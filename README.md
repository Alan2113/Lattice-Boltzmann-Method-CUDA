# LBM — Lattice Boltzmann Method na GPU

Symulacja **dyfuzji** metodą **Lattice Boltzmann** w schemacie **D2Q4**, zaimplementowana w CUDA z wizualizacją OpenGL w czasie rzeczywistym.

![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-supported-blue?logo=cplusplus&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-FreeGLUT%20%2B%20GLEW-5586A4?logo=opengl)

## Spis treści

- [Opis modelu](#opis-modelu)
- [Algorytm LBM](#algorytm-lbm)
- [Wymagania](#wymagania)
- [Instalacja bibliotek](#instalacja-bibliotek)
- [Budowanie](#budowanie)
- [Uruchomienie](#uruchomienie)
- [Sterowanie](#sterowanie)
- [Konfiguracja](#konfiguracja)
- [Implementacja](#implementacja)

---

## Opis modelu

**Lattice Boltzmann Method (LBM)** to nowoczesna metoda symulacji przepływów i procesów transportu, oparta na dyskretyzacji równania Boltzmanna na siatce. Zamiast śledzić indywidualne cząstki (jak w LGA), LBM operuje na **funkcjach rozkładu** `f_i(r, t)` — gęstościach prawdopodobieństwa znalezienia cząstki w danym punkcie z daną prędkością.

### Wariant D2Q4 — dyfuzja

Schemat **D2Q4** (2 wymiary, 4 kierunki prędkości) jest najprostszym modelem LBM, idealnym do **modelowania dyfuzji** (np. rozprzestrzeniania się stężenia substancji, ciepła w cieczy w spoczynku).

Każda komórka przechowuje 4 funkcje rozkładu odpowiadające kierunkom:

```
        f₂ (góra)
         ↑
f₃ ← ── ● ── → f₁
         ↓
        f₄ (dół)
```

### Scenariusz symulacji

- Siatka **128×128** komórek otoczona ścianami
- Pionowa **bariera w środku** z otworem 40 komórek
- Lewa strona bariery: stężenie **C = 1.0**
- Prawa strona: stężenie **C = 0.0**
- Obserwujemy dyfuzję przez otwór i wyrównywanie stężeń

## Algorytm LBM

W każdym kroku czasowym wykonywane są 4 etapy:

### 1. Obliczenie stężenia
Stężenie `C` w komórce to suma funkcji rozkładu (wzór 9):

```
C(r, t) = Σ f_i(r, t)
```

### 2. Obliczenie rozkładu równowagowego
Dla D2Q4 rozkład równowagowy (wzór 10):

```
f_eq_i = w_i · C,    gdzie w_i = 1/4 dla każdego kierunku
```

### 3. Kolizja (operator BGK)
Relaksacja w stronę równowagi z czasem `τ` (wzór 11):

```
f_out_i = f_in_i + (Δt/τ) · (f_eq_i - f_in_i)
```

### 4. Streaming
Przeniesienie funkcji rozkładu do sąsiednich komórek (wzór 12a, *pull strategy*):

```
f_in(r, t+Δt) = f_out(r - c_i, t)
```

Na ścianach stosowane jest **bounce-back** — odbicie funkcji rozkładu (wzór 13).

## Wymagania

| Komponent | Wymaganie |
|-----------|-----------|
| **CUDA Toolkit** | wymagany (z nvcc) |
| **CMake** | 3.24+ |
| **Kompilator C++** | MSVC 2019/2022 (Visual Studio Build Tools) |
| **Karta graficzna** | NVIDIA z Compute Capability ≥ 7.5 |
| **System** | Windows x64 |

> **Uwaga:** Architektura CUDA w `CMakeLists.txt` ustawiona jest na `89` (RTX 4070, Ada Lovelace). Dla innych kart zmień wartość — np. `86` dla RTX 30xx, `75` dla RTX 20xx.

## Instalacja bibliotek

Repozytorium nie zawiera bibliotek graficznych — pobierz je ręcznie do folderu `libs/`.

### freeglut 3.4.0
Pobierz wersję MSVC binary z https://www.transmissionzero.co.uk/software/freeglut-devel/ i rozpakuj jako `libs/freeglut/`.

### GLEW 2.2.0
Pobierz Windows binary z https://glew.sourceforge.net/ i rozpakuj jako `libs/glew-2.2.0/`.

### Wynikowa struktura

```
2.LBM/
├── CMakeLists.txt
├── main.cu
├── libs/
│   ├── freeglut/
│   │   ├── include/GL/
│   │   ├── lib/x64/freeglut.lib
│   │   └── bin/x64/freeglut.dll
│   └── glew-2.2.0/
│       ├── include/GL/
│       ├── lib/Release/x64/glew32.lib
│       └── bin/Release/x64/glew32.dll
└── README.md
```

## Budowanie

### Z CLion

1. **File → Open** → wskaż folder `2.LBM`
2. **Settings → Build → CMake** — ustaw toolchain na **Visual Studio**
3. **Tools → CMake → Reset Cache and Reload Project**
4. Wybierz target **LBM** z dropdown'a obok przycisku Run i zbuduj (Ctrl+F9)

### Z linii poleceń (PowerShell / cmd)

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target LBM
```

## Uruchomienie

```cmd
build\LBM.exe
```

CMake automatycznie kopiuje wymagane pliki DLL (`freeglut.dll`, `glew32.dll`) obok pliku wykonywalnego, więc aplikacja uruchamia się od razu bez ręcznego kopiowania.

## Sterowanie

| Klawisz | Akcja |
|---------|-------|
| `Spacja` | Pauza / wznowienie |
| `+` lub `=` | Przyspieszenie symulacji |
| `-` lub `_` | Zwolnienie symulacji |
| `R` | Reset symulacji |
| `Esc` | Wyjście |

## Konfiguracja

Parametry modyfikowalne na początku pliku `main.cu`:

```cpp
const int Nx = 128, Ny = 128;     // rozmiar siatki
const int WINDOW_SIZE = 512;       // rozmiar okna w pikselach
const float TAU = 1.0f;            // czas relaksacji τ
```

### Wpływ czasu relaksacji τ

Parametr `τ` kontroluje szybkość dyfuzji:

- **τ = 1.0** — pełne dążenie do równowagi w jednym kroku (szybka dyfuzja)
- **τ > 1.0** — wolniejsza dyfuzja, większa stabilność numeryczna
- **τ < 0.5** — niestabilność numeryczna (unikać)

Współczynnik dyfuzji jest powiązany z τ wzorem `D = (τ - 0.5) / 4` (dla D2Q4 z `Δt = 1`).

## Implementacja

### Kernele CUDA

W każdym kroku symulacji uruchamiane są 4 kernele odpowiadające etapom algorytmu:

- **`computeConcentration`** — sumuje funkcje rozkładu, oblicza stężenie `C`
- **`computeEquilibrium`** — wyznacza rozkład równowagowy `f_eq` z stężenia
- **`collision`** — operator BGK: relaksacja `f_in → f_eq` z czasem `τ`
- **`streaming`** — przeniesienie funkcji rozkładu do sąsiadów (PULL strategy z bounce-back na ścianach)

### Layout pamięci

Funkcje rozkładu przechowywane są jako **`float4`** (4 × 32 bit) — jedna komórka = jeden odczyt/zapis pamięci GPU. Trzy bufory (`d_f_in`, `d_f_eq`, `d_f_out`) zgodnie ze strukturą algorytmu z materiałów kursu.

### Wizualizacja

Każda komórka renderowana jest jako punkt OpenGL (`GL_POINTS` o rozmiarze 4 px) z gradientem barwnym:

- **C = 1.0** → fioletowy (wysokie stężenie)
- **C = 0.5** → niebieski (stan przejściowy)
- **C = 0.0** → zielony (niskie stężenie)

Ściany rysowane są na ciemno, żeby nie zaburzały odczytu pola stężenia.
