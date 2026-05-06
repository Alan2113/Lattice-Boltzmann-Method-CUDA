#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ========== PARAMETRY ==========
const int Nx = 128, Ny = 128;  // Większa siatka dla lepszej wizualizacji
const int WINDOW_SIZE = 512;   // Większe okno
const float TAU = 1.0f;  // Czas relaksacji τ

// ========== GPU - 3 ZESTAWY FUNKCJI ROZKŁADU (zgodnie z PDF) ==========
float4 *d_f_in;   // Funkcje wejściowe f_in
float4 *d_f_eq;   // Funkcje równowagowe f_eq
float4 *d_f_out;  // Funkcje wyjściowe f_out
float *d_C;       // Stężenie C
int *d_walls;     // Ściany

// ========== CPU ==========
float *h_C;       // Stężenie do wizualizacji
int *h_walls;     // Mapa ścian

// ========== STAN SYMULACJI ==========
int step = 0;
bool paused = false;
int speed_factor = 3;  // Wolniej dla większej siatki

// ========== WAGI D2Q4 ==========
// Wzór (10): wi = 0.25 dla wszystkich kierunków w D2Q4
__constant__ float w[4] = {0.25f, 0.25f, 0.25f, 0.25f};

// ========================================================================
// KERNEL 1: Obliczenie stężenia C
// Wzór (9): C = Σ f_i (suma wszystkich składowych)
// ========================================================================
__global__ void computeConcentration(float4 *f_in, float *C, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;
    float4 f = f_in[idx];

    // Wzór (9): C = f1 + f2 + f3 + f4
    C[idx] = f.x + f.y + f.z + f.w;
}

// ========================================================================
// KERNEL 2: Obliczenie rozkładu równowagowego f_eq
// Wzór (10): f_eq_i = w_i * C
// ========================================================================
__global__ void computeEquilibrium(float *C, float4 *f_eq, int *walls, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;

    // Na ścianach f_eq = 0
    if (walls[idx]) {
        f_eq[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float c = C[idx];

    // Wzór (10): f_eq_i = w_i * C
    // Dla D2Q4: w1 = w2 = w3 = w4 = 0.25
    f_eq[idx] = make_float4(
        w[0] * c,  // f_eq_1 = 0.25 * C
        w[1] * c,  // f_eq_2 = 0.25 * C
        w[2] * c,  // f_eq_3 = 0.25 * C
        w[3] * c   // f_eq_4 = 0.25 * C
    );
}

// ========================================================================
// KERNEL 3: Kolizja - obliczenie f_out
// Wzór (11): f_out_i = f_in_i + (Δt/τ) * (f_eq_i - f_in_i)
// ========================================================================
__global__ void collision(float4 *f_in, float4 *f_eq, float4 *f_out,
                          int *walls, float tau, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;

    // Na ścianach f_out = 0
    if (walls[idx]) {
        f_out[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 fin = f_in[idx];
    float4 feq = f_eq[idx];

    // Wzór (11): f_out = f_in + (Δt/τ) * (f_eq - f_in)
    // Δt = 1 (krok czasowy)
    float omega = 1.0f / tau;

    f_out[idx] = make_float4(
        fin.x + omega * (feq.x - fin.x),
        fin.y + omega * (feq.y - fin.y),
        fin.z + omega * (feq.z - fin.z),
        fin.w + omega * (feq.w - fin.w)
    );
}

// ========================================================================
// KERNEL 4: Streaming - przeniesienie f do sąsiadów
// Wzór (12): f_in(r+c_i, t+Δt) = f_out(r, t)
// lub (12a): f_in(r, t+Δt) = f_out(r-c_i, t)
//
// Model prędkości D2Q4:
// c1 = {1, 0}  - prawo
// c2 = {0, 1}  - góra
// c3 = {-1, 0} - lewo
// c4 = {0, -1} - dół
// ========================================================================
__global__ void streaming(float4 *f_out, float4 *f_in, int *walls, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;

    // Ściany
    if (walls[idx]) {
        f_in[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // PULL strategy (wzór 12a): f_in(r) = f_out(r - c_i)

    // f1 (kierunek prawo c1={1,0}): przychodzi z lewej (x-1, y)
    float f1 = 0.0f;
    if (x > 0) {
        int idx_left = (x-1) + y * Nx;
        if (!walls[idx_left]) {
            f1 = f_out[idx_left].x;
        } else {
            // Wzór (13): bounce-back - odbicie od ściany
            f1 = f_out[idx].z;
        }
    }

    // f2 (kierunek góra c2={0,1}): przychodzi z dołu (x, y-1)
    float f2 = 0.0f;
    if (y > 0) {
        int idx_down = x + (y-1) * Nx;
        if (!walls[idx_down]) {
            f2 = f_out[idx_down].y;
        } else {
            f2 = f_out[idx].w;
        }
    }

    // f3 (kierunek lewo c3={-1,0}): przychodzi z prawej (x+1, y)
    float f3 = 0.0f;
    if (x < Nx - 1) {
        int idx_right = (x+1) + y * Nx;
        if (!walls[idx_right]) {
            f3 = f_out[idx_right].z;
        } else {
            f3 = f_out[idx].x;
        }
    }

    // f4 (kierunek dół c4={0,-1}): przychodzi z góry (x, y+1)
    float f4 = 0.0f;
    if (y < Ny - 1) {
        int idx_up = x + (y+1) * Nx;
        if (!walls[idx_up]) {
            f4 = f_out[idx_up].w;
        } else {
            f4 = f_out[idx].y;
        }
    }

    f_in[idx] = make_float4(f1, f2, f3, f4);
}

// ========================================================================
// INICJALIZACJA
// ========================================================================
void initLBM() {
    size_t size_f = Nx * Ny * sizeof(float4);
    size_t size_c = Nx * Ny * sizeof(float);
    size_t size_w = Nx * Ny * sizeof(int);

    // Alokacja GPU - 3 zestawy funkcji rozkładu (zgodnie z PDF)
    cudaMalloc(&d_f_in, size_f);
    cudaMalloc(&d_f_eq, size_f);
    cudaMalloc(&d_f_out, size_f);
    cudaMalloc(&d_C, size_c);
    cudaMalloc(&d_walls, size_w);

    // Alokacja CPU
    h_C = (float*)malloc(size_c);
    h_walls = (int*)calloc(Nx * Ny, sizeof(int));

    // === Ściany ===
    // Ramka wokół
    for (int x = 0; x < Nx; x++) {
        h_walls[x + 0 * Nx] = 1;           // dół
        h_walls[x + (Ny-1) * Nx] = 1;      // góra
    }
    for (int y = 0; y < Ny; y++) {
        h_walls[0 + y * Nx] = 1;           // lewo
        h_walls[(Nx-1) + y * Nx] = 1;      // prawo
    }

    // Bariera z większym otworem na środku
    int barrier_x = Nx / 2;
    int gap_size = 40;  // Większy otwór dla lepszej wizualizacji
    for (int y = 0; y < Ny; y++) {
        if (y < Ny/2 - gap_size/2 || y > Ny/2 + gap_size/2) {
            h_walls[barrier_x + y * Nx] = 1;
        }
    }

    cudaMemcpy(d_walls, h_walls, size_w, cudaMemcpyHostToDevice);

    // === Warunki początkowe (zgodnie z PDF) ===
    // Lewa część: C = 1.0
    // Prawa część: C = 0.0

    float4 *h_f_in = (float4*)calloc(Nx * Ny, sizeof(float4));

    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            int idx = x + y * Nx;

            if (h_walls[idx]) continue;

            // Lewa strona bariery: C = 1.0
            if (x < barrier_x - 5) {
                h_f_in[idx] = make_float4(0.25f, 0.25f, 0.25f, 0.25f);
            }
            // Prawa strona: C = 0.0 (już wyzerowane)
        }
    }

    cudaMemcpy(d_f_in, h_f_in, size_f, cudaMemcpyHostToDevice);
    free(h_f_in);

    printf("=== LBM Dyfuzja - Inicjalizacja ===\n");
    printf("Siatka: %dx%d\n", Nx, Ny);
    printf("Schemat: D2Q4\n");
    printf("Bariera: x=%d, otwor: %d komorek\n", barrier_x, gap_size);
    printf("Czas relaksacji τ: %.1f\n", TAU);
    printf("Warunki poczatkowe:\n");
    printf("  - Lewa strona (x<%d): C=1.0\n", barrier_x);
    printf("  - Prawa strona (x>%d): C=0.0\n", barrier_x);
    printf("===================================\n\n");
}

// ========================================================================
// SYMULACJA - jeden krok czasowy
// ========================================================================
void simulate() {
    if (paused) return;

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    // Krok 1: Oblicz stężenie C (wzór 9)
    computeConcentration<<<grid, block>>>(d_f_in, d_C, Nx, Ny);

    // Krok 2: Oblicz rozkład równowagowy f_eq (wzór 10)
    computeEquilibrium<<<grid, block>>>(d_C, d_f_eq, d_walls, Nx, Ny);

    // Krok 3: Kolizja - oblicz f_out (wzór 11)
    collision<<<grid, block>>>(d_f_in, d_f_eq, d_f_out, d_walls, TAU, Nx, Ny);

    // Krok 4: Streaming - przenieś f_out → f_in (wzór 12)
    streaming<<<grid, block>>>(d_f_out, d_f_in, d_walls, Nx, Ny);

    step++;
}

// ========================================================================
// WIZUALIZACJA Z LEPSZYM GRADIENTEM
// ========================================================================
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Pobierz stężenie C z GPU
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    computeConcentration<<<grid, block>>>(d_f_in, d_C, Nx, Ny);
    cudaMemcpy(h_C, d_C, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);

    // Rysuj
    glPointSize(4.0f);  // Mniejsze punkty dla większej siatki
    glBegin(GL_POINTS);

    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            int idx = x + y * Nx;

            if (h_walls[idx]) {
                // Ściany - czarny (żeby nie przeszkadzały)
                glColor3f(0.1f, 0.1f, 0.1f);
                glVertex2f(x + 0.5f, y + 0.5f);
            } else {
                float c = h_C[idx];

                // NOWY GRADIENT: Fioletowy → Zielony
                // C=1.0 → fioletowy (RGB: 0.8, 0, 1)
                // C=0.5 → niebieski (RGB: 0, 0.5, 1)
                // C=0.0 → zielony (RGB: 0, 1, 0.2)

                float r = 0.8f * c;           // R: 0.8→0
                float g = 1.0f - c*0.8f;      // G: 0.2→1
                float b = 1.0f - c*0.2f;      // B: 0.8→1

                glColor3f(r, g, b);
                glVertex2f(x + 0.5f, y + 0.5f);
            }
        }
    }

    glEnd();

    // Info - większa czcionka, biały tekst
    char info[150];
    sprintf(info, "Krok: %d %s | v:%d | Siatka: %dx%d",
            step, paused ? "[PAUZA]" : "", speed_factor, Nx, Ny);
    glColor3f(0.0f, 0.7f, 1.0f);  // Biały tekst
    glRasterPos2f(7, Ny - 10);
    for (char *c = info; *c; c++)
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *c);

    glutSwapBuffers();
}

void idle() {
    static int skip = 0;
    if (++skip < speed_factor) {
        glutPostRedisplay();
        return;
    }
    skip = 0;

    simulate();
    glutPostRedisplay();

    if (step % 100 == 0 && !paused)
        printf("Krok: %d\n", step);
}

void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case ' ':
            paused = !paused;
            printf("%s\n", paused ? "PAUZA" : "WZNOWIONO");
            break;
        case '+': case '=':
            if (speed_factor > 1) speed_factor--;
            printf("Predksc: %d\n", speed_factor);
            break;
        case '-': case '_':
            if (speed_factor < 10) speed_factor++;
            printf("Predkosc: %d\n", speed_factor);
            break;
        case 'r': case 'R':
            cudaFree(d_f_in);
            cudaFree(d_f_eq);
            cudaFree(d_f_out);
            cudaFree(d_C);
            cudaFree(d_walls);
            free(h_C);
            free(h_walls);
            step = 0;
            paused = false;
            initLBM();
            printf("RESET - nowa symulacja\n");
            break;
        case 27:
            exit(0);
    }
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, Nx, 0, Ny);
    glMatrixMode(GL_MODELVIEW);
}

// ========================================================================
// MAIN
// ========================================================================
int main(int argc, char** argv) {
    // Wykryj GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\n=== GPU INFO ===\n");
        printf("Karta: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Multiprocesory: %d\n", prop.multiProcessorCount);
        printf("Pamiec: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
        printf("================\n\n");
    } else {
        printf("BLAD: Nie znaleziono GPU CUDA!\n");
        return 1;
    }

    // Inicjalizacja OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE);
    glutCreateWindow("LBM - Modelowanie Dyfuzji (D2Q4)");
    glewInit();

    initLBM();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    printf("=== STEROWANIE ===\n");
    printf("SPACJA - pauza/wznow\n");
    printf("+/-    - zmiana predkosci\n");
    printf("R      - reset symulacji\n");
    printf("ESC    - wyjscie\n");
    printf("==================\n\n");

    glutMainLoop();
    return 0;
}
