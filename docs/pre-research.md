# Rust製原子核反応率計算フレームワークの技術要件調査

宇宙核天体物理向けHauser-Feshbach/R-matrix計算エンジンをRustで新規構築することは**技術的に実現可能であり、既存コードに対して50〜500倍のGPU高速化が見込める**。既存のFortranコード群（TALYS, SMARAGD等）はすべてシリアル実行であり、GPU並列化されたHFコードは世界に存在しない。Rustエコシステムは2025年時点で科学計算に必要な主要ライブラリ（faer線形代数、cudarc GPU、rayon並列化、PyO3 Python連携）が実用段階に達しており、この空白を埋める絶好の機会である。本調査では既存コードのボトルネック分析からRustクレート選定、GPU並列化戦略、OSS設計パターンまでを網羅的にまとめた。

---

## 既存コードはシリアル実行のFortran/C++に依存している

### TALYSの構造と計算ボトルネック

TALYS（v2.2、Fortran 95、MITライセンス、GitHub: `arjankoning1/talys`）は核反応計算の事実上の標準コードである。構造データベースを含め**約8GB**のディスク容量を占め、中央モジュール`A0_talys_mod.f90`にグローバル変数を集約するフラットなモジュール構成をとる。計算パイプラインは入力解析→核構造データ読込→光学模型/結合チャネル→直接反応→前平衡→Hauser-Feshbach複合核→多重粒子放出→出力の順に逐次実行される。

**最大のボトルネックはECIS-06結合チャネル光学模型計算**である。J. Raynalの結合チャネルコードをサブルーチン化して内蔵しており、変形核（アクチノイド等）では結合状態数²×部分波数×エネルギー点数に比例して計算量が爆発する。AMDOpteron上でサンプル問題の実行に約**1時間**を要し、TENDL評価済み核データライブラリの生成では1核種あたり数千回のモンテカルロ実行が必要になる。ニューラルネットワーク代理モデルがTALYS-2.0に対して**1000倍以上**の高速化を達成した報告（arXiv:2602.21239, 2026年2月）は、ベースラインコードの計算コストの大きさを物語る。

### AZURE2のR-matrixカーネル

AZURE2（C++、GPL-3.0、GitHub: `rdeboer1/AZURE2`）はLane-Thomas R-matrix形式を実装し、GSLとMinuit2に依存する。中核の計算パターンはチャネル行列 **(1 − RL₀)** の各エネルギー点における逆行列計算であり、「多チャネル・少レベル」の場合にこの行列反転が主要ボトルネックとなる。χ²フィッティングではMinuit2が数千回のコスト関数評価を行い、各評価で全実験エネルギー点についてR-matrix計算をフル実行する。OpenMP並列化はMinuit2経由でのみ利用可能だが、ROOTビルドではデフォルトで無効であるため、性能が大幅に低下するケースがある。典型的なフィッティングは**9レベル、3粒子対、約16パラメータ**規模（Frontiers in Physics, 2022）。

### CoH3とEMPIRE

CoH3（C++、約200ソース/ヘッダファイル、GitHub: `toshihikokawano/coh3`、LANL）はTALYSより軽量なC++実装で、Engelbrecht-Weidenmüller変換によるオフ対角S行列要素の適切な処理が特徴。EMPIRE（Fortran 77、BNL）はECIS-06、SCAT2、ORION等を統合した大規模コードで、EMPIRE-IIリファクタリング時にスクラッチファイル除去により**約20倍の高速化**を達成した歴史がある。

### SMARAGD：天体物理特化のHFコード

SMARAGD（v0.42.0s、Fortran、非公開）はNON-SMOKER（Rauscher & Thielemann, 2000）の後継として天体物理反応率計算に特化している。Fox-Goodwinアルゴリズムと改良Coulomb波動関数ルーチンにより、**Coulomb障壁よりはるか下方でも数値的に安定な透過係数計算**を実現した（arXiv:2512.03206）。NON-SMOKERデータベースはZ=9〜83の数千核種をカバーする。

### 全コード共通のボトルネック

すべてのコードに共通する4つの計算ボトルネックが存在する。第一に、光学模型ポテンシャルからの**透過係数計算**（部分波・エネルギー点ごとのSchrödinger方程式数値積分）。第二に、**複素Coulomb関数の高精度評価**（Coulomb障壁以下で数十桁の精度損失が発生）。第三に、HF合算における**スピン・パリティ全組合せの多重総和**（J_max × N_channels × N_energy_bins × N_residual_nuclides のスケーリング）。第四に、Maxwell平均断面積の**温度×エネルギー二重ループ**（20〜60温度点 × 50〜200エネルギー点 × 熱的励起状態数）。いずれも**完全にシリアル実行**されており、並列化による巨大な高速化余地がある。

---

## Rustの科学計算エコシステムは実用段階に到達した

### 線形代数：faerが最適解

密行列計算には**faer**（v0.23.2、GitHub: `sarah-quinones/faer-rs`、MIT/Apache-2.0）を推奨する。純RustでBLAS/LAPACK外部依存なしに、1024×1024行列乗算でfaer(par)=**8.2ms**に対しndarray(BLAS)=9.6ms、nalgebra=39ms、C++ Eigen=8.3msという性能を達成する（Intel i5-11400, 12スレッド）。LU分解・Cholesky分解・SVD・固有値分解をすべてサポートし、複素数対応、疎行列モジュール（`faer::sparse`）、呼び出しごとの並列度制御機能を備える。

| ライブラリ | 1024×1024 matmul | Cholesky | 特徴 |
|-----------|-----------------|----------|------|
| faer(par) | 8.2ms | 3.3ms | 純Rust、並列、最高性能 |
| ndarray+BLAS | 9.6ms | 15ms | NumPy互換、Python連携 |
| nalgebra | 39ms | 44ms | 小行列に強い、静的サイズ |
| Eigen (C++) | 8.3ms | 8.2ms | 参考値 |

疎行列は`faer::sparse`（高性能ソルバ付き）と`nalgebra-sparse`（CSR/CSC/COO）を組み合わせる。ndarray（v0.16）はPython連携（`rust-numpy`によるNumPyとのゼロコピー橋渡し）が必要な場合に使用する。

### 特殊関数：Coulomb関数がクリティカルギャップ

核散乱計算に不可欠な**Coulomb関数を提供するRust純正クレートは存在しない**。唯一の選択肢はGSLバインディングの**rgsl**（GitHub: `GuillaumeGomez/rust-GSL`）で、正則・不正則Coulomb波動関数、超幾何関数、ガンマ関数を含むGSL全機能にアクセスできるが、libgslシステム依存とFFIオーバーヘッドが課題である。一般的なBessel関数には**puruspe**（v0.4.1、純Rust、月28K DL）、複素引数Bessel関数にはAmos(TOMS 644)アルゴリズムの純Rust実装**complex-bessel**が利用可能。**Coulomb関数の純Rust高性能実装はこのプロジェクトで開発すべき最重要サブコンポーネント**である。

### GPU計算：cudarcが最も成熟

Rust GPUコンピューティングの現状（2025-2026年）は以下の通り：

- **cudarc**（v0.19.3、GitHub: `coreylowman/cudarc`、全DL数119万）：CUDA Driver API、NVRTC（ランタイムコンパイル）、cuBLAS、cuRANDの安全なラッパー。BurnMLフレームワークの基盤として実戦投入済み。動的ロードによりビルド時のCUDA不要。**ホスト側CUDA管理の第一選択**
- **Rust-CUDA**（GitHub: `Rust-GPU/Rust-CUDA`）：2025年1月にリブート、`rustc_codegen_nvvm`バックエンドでRustコードをPTXにコンパイル。2025年7月に同一RustカーネルをCPU/SPIR-V/CUDAで実行するデモを達成。まだプロダクション未対応
- **wgpu**：クロスプラットフォームGPU計算だが、**WGSLがf64非対応**のため核物理計算には不適（SPIR-V拡張経由では可能だが制約が大きい）
- **CubeCL**（v0.8.1）：Burnが使用する計算抽象レイヤー。CUDA/wgpu上の抽象化

**推奨戦略**：cudarcでホスト側管理し、カーネルはCUDA Cで記述（PTXをランタイムコンパイル）。将来的にRust-CUDAが成熟すれば純Rustカーネルに移行する。

### SIMD：std::simdは未安定、wideクレートを使用

`std::simd`（Portable SIMD）は**2026年3月現在もnightly限定で安定化されていない**。`LaneCount<N>: SupportedLaneCount`制約やSwizzle APIの人間工学的問題が未解決（GitHub issue #364）。安定版Rustで利用可能な選択肢として**wide**（v0.7.x、`f64x4`等を提供）を推奨する。Rust 1.87以降、`std::arch`の大半のSIMD組込み関数がunsafe不要になった点は有利。**Coulomb関数のSIMDバッチ計算は技術的に実現可能**だが、パラメータ依存の収束回数の違い（ワープダイバージェンス相当）への対策が必要で、パラメータ領域別にビンニングして処理する方式が有効。

### 並列化とMPI

**rayon**（v1.10、ワークスティーリングスレッドプール）はRustにおけるデータ並列の標準。NAS Parallel Benchmarks（arXiv:2502.15536, 2025）によれば、逐次実行でFortranの**約1.2%遅い**、C++より**約5.6%速い**。40スレッド並列ではFortran+OpenMPの約**16%遅い**。`nowait`相当がないことが主な差因。分散計算には**rsmpi**（v0.8.0、MPI 3.1準拠、OpenMPI/MPICH/MS-MPI対応）が利用可能で、ポイントツーポイント・集団通信・カスタムリダクションをサポートする。片側通信（RMA）とMPI-IOは未対応。

### Python/Fortran連携

**PyO3**（v0.27）+ **rust-numpy** + **maturin**によるPythonバインディングが最も成熟している。NumPy配列とのゼロコピー連携が可能で、pynucastro等の天体物理ネットワークコードとの統合に最適。Fortran FFIは`extern "C"` + Fortran側`bind(C, name="...")`で実現し、`cc`クレートから`build.rs`でgfortranを呼び出して既存TALYS/AZURE2サブルーチンを静的ライブラリとしてリンクできる。

---

## GPU並列化で50〜500倍の高速化が見込める

### 光学模型のバッチODE並列化が最大のインパクト

透過係数計算の核心であるNumerov法（4次精度差分法）は**逐次的に動径座標を積分する**ため、単一インスタンスの並列化は困難だが、バッチ並列化は極めて効果的である。1000エネルギー点 × 30部分波 = **30,000独立Numerov積分**がGPUスレッドに直接マッピングされる。1積分あたりのメモリは約16KB（L1/共有メモリに収まる）、演算量は約20,000FLOPで、現代GPUには軽微なワークロード。

DiffEqGPU.jl（Julia SciML）のベンチマークでは、同一ODEの異なるパラメータでのアンサンブル求解で**100〜1000倍の高速化**を達成している。核物理のバッチODE（同じNumerov法、異なるエネルギー/部分波/核種）は本質的に同じ構造であり、**50〜500倍のGPU高速化が現実的**である。MPGOS（CUDA C++）フレームワークがRK4/RK-Cash-Karp法のGPUバッチ実装を提供しており、設計参考になる。

### HF合算のGPUカーネル化

HF断面積公式のJπ合算は**map-reduce演算としてGPU化**できる。外側のJπループ（J=0〜30、π=±1、計約60項）と内側のチャネル合算を2次元GPUスレッドグリッドにマッピングし、分母の全透過係数総和はワープシャッフルと共有メモリリダクションで計算する。全核図バッチ計算（10,000核種対 × 50温度 × 500エネルギー = **2.5億独立評価**）はGPUに理想的なワークロード。

### R-matrix計算：HPRMAT前例がある

**HPRMAT**（Lei, 2024, arXiv:2512.11590）は核物理向けGPU R-matrixソルバーの唯一の公開実装である。RTX 3090上でN=25,600行列に対し最適化CPU直接解法の**9倍**、レガシー逆行列法の**18倍**の高速化を達成した。コンシューマGPUのFP64:FP32スループット比（64:1）を活用した**混合精度戦略**（FP32でLU分解→反復精緻化でFP64精度を回復）が鍵であり、断面積の相対誤差**10⁻⁵以下**を維持する。

χ²フィッティングのGPU加速にはGpufit（Scientific Reports 7, 2017）が参考になり、Levenberg-Marquardt法でGTX 1080上**42倍**の高速化、最大スループット**465万fits/秒**を達成している。MCMC（BRICK/emcee型）ではウォーカー間の並列尤度評価が自然にGPUマッピングされる。

### Maxwell平均断面積の並列化

MACS積分はGauss-Laguerre求積法で効率的に計算される。15〜20点の求積ノードと重みは定数としてGPUコンスタントメモリに格納し、各GPUスレッドが1つの（核種、温度）組合せの重み付き和を計算する。積分カーネル評価（各ノードエネルギーでの断面積計算）が主コストであり、光学模型→透過係数→HF合算→MACS積分の**エンドツーエンドGPUパイプライン**によりデータ転送オーバーヘッドを最小化すべき。

---

## OSS設計はBurnパターンとCargoワークスペースが基盤

### 推奨クレート構成

成功しているRust科学計算OSS（Polars: 20+クレート、Burn: バックエンド抽象化、faer: 純Rust高性能）の共通パターンに基づき、以下の階層的ワークスペース構成を推奨する：

```
nucrust/
├── Cargo.toml                      # ワークスペースルート
├── crates/
│   ├── nucrust-core/               # 型定義、単位系、エラー処理
│   ├── nucrust-data/               # RIPL-3/REACLIB/ENDFパーサー
│   ├── nucrust-special/            # Coulomb関数、Bessel関数（純Rust）
│   ├── nucrust-optical/            # 光学模型（Numerov、結合チャネル）
│   ├── nucrust-hf/                 # Hauser-Feshbach統計模型
│   ├── nucrust-rmatrix/            # R-matrix理論
│   ├── nucrust-astro/              # 天体物理反応率（MACS、恒星反応率）
│   ├── nucrust-gpu/                # GPUバックエンド（cudarc）
│   ├── nucrust-python/             # PyO3バインディング
│   └── nucrust/                    # 集約クレート（feature flags）
├── benches/                        # Criterionベンチマーク
├── book/                           # mdbookユーザーガイド
└── tests/reference_data/           # TALYS/AZURE2検証用ゴールデンファイル
```

### バックエンド抽象化パターン

Burnの`Backend`トレイトパターンを採用し、CPU/GPU実装をコンパイル時に切り替える：

```rust
pub trait ComputeBackend {
    fn batch_numerov(params: &NumerovParams) -> TransmissionCoeffs;
    fn hf_summation(tc: &TransmissionCoeffs, ld: &LevelDensity) -> CrossSection;
    fn rmatrix_solve(rmat: &RMatrix, energies: &[f64]) -> CollisionMatrix;
}
pub struct CpuBackend;  // rayon + faer
pub struct GpuBackend;  // cudarc + CUDA Cカーネル
```

feature flagsで`gpu = ["dep:cudarc", "nucrust-gpu"]`、`parallel = ["dep:rayon"]`、`simd = []`、`hdf5 = ["dep:hdf5"]`等を制御する。

### テスト・検証戦略

5層のテスト階層を設計する。(1) 単体テスト：1チャネルR-matrix解析解、単純光学模型等との比較。(2) 統合テスト：TALYS/AZURE2出力とのゴールデンファイル比較（`approx`クレートで`assert_relative_eq!`、ε=10⁻⁶）。(3) 性質ベーステスト：`proptest`で断面積≥0、ユニタリ性、詳細釣合を検証。(4) 回帰テスト：Criterion.rs（v0.5）でベンチマーク回帰を検出し、Iai（Cachegrind命令計数）でCI環境のVM揺らぎを排除。(5) ファズテスト：RIPL-3/REACLIBパーサーのランダム入力耐性。

### データ形式の実装要件

**RIPL-3**（IAEA、2009年公開）は7セグメント構成の固定幅テキストファイルで、質量・離散準位・共鳴パラメータ・光学模型パラメータ・レベル密度・γ線強度・核分裂障壁を含む約9000核種分のデータを格納する。Fortranフォーマット文字列（`FORMAT(I3,1X,A5,...)`型）の厳密なパース実装が必要。**REACLIB**（JINA）は厳密に74文字/行の固定幅フォーマットで、3行1セットの構造を持ち、7パラメータ（a₀〜a₆）の反応率公式係数を格納する。チャプター番号（1-11）で反応トポロジーを定義する。HDF5対応には**hdf5-metno**（`metno/hdf5-rust`フォーク、Rust≥1.80、ndarray連携、LZF/Blosc圧縮対応）を推奨。構造化データI/Oにはserdeエコシステム（TOML設定、MessagePack高速バイナリ、JSONデバッグ用）を使用する。

---

## 先行Rustプロジェクトから得られる教訓

### 量子化学QSym²が示すモデルケース

**QSym²**（GitLab: `bangconghuynh/qsym2`、JCTC 2023掲載）はRust科学計算OSSの最も成功した事例の一つである。Pythonプロトタイプ（poly-inspect）からRustに完全リライトし、ndarray-lalg + 設定可能なBLASバックエンド（OpenBLAS/LAPACK）を採用。Q-Chem、Orca、QUEST等の既存量子化学コードとのHDF5/ファイル形式インターフェースによる統合を実現した。Rustの型システムを活用した汎用的な対称性解析が設計の核心であり、ハードコードされた制限を排除した。

**REST**（Gitee: `RESTGroup/rest`、GPL v2）はRust-Fortranハイブリッドアプローチの実例で、高レベルロジックをRust、最適化テンソル演算をFortran（`restmatr.f90`）で実装する。汎用ndarray型ではなくDFT/WFT計算パターンに特化した独自テンソル型（`MatrixFull`、`RIFull`、`ERIFull`）を定義している点は、核物理フレームワークでも光学模型行列やR-matrix用の特殊型を検討する根拠となる。

### 分子動力学lumolのトレイト設計

**lumol**（GitHub: `lumol-org/lumol`、206 stars、BSD 3-clause）はトレイトベースの拡張性設計（ポテンシャル、積分器、MC移動にそれぞれトレイトを定義）を採用。同プロジェクトの**soa-derive**クレート（470 stars）は構造体のStruct-of-Arrays変換を自動化し、粒子データのキャッシュ効率を大幅に改善する。核物理計算でもエネルギーグリッド上の断面積データ配列にSoAパターンが有効。一方、GPU対応はRustエコシステムの未成熟を理由に見送られた（2020年時点）。

### 共通の課題と対策

先行プロジェクトから抽出される課題は3点に集約される。**GPU対応の欠如**が最大のギャップで、事実上すべてのRust科学計算プロジェクトで意味のあるGPU加速が実装されていない。**単一開発者への依存**（lumol、velvet等）がサステナビリティリスクであり、コミュニティ形成が不可欠。**BLAS/LAPACK FFI依存**が多くのプロジェクトで見られたが、faerの登場により純Rust代替が実用段階に達した。CBET物理シミュレーション（arXiv:2410.19146, ロチェスター大学）ではRustがC++に対し最大**5.6倍の高速化**を達成した報告もあり、Rustの数値計算性能は実証されている。

---

## 性能目標と現実的な高速化見積もり

### Rust対Fortran/C++の基本性能

NAS Parallel Benchmarks（arXiv:2502.15536, 2025）による最新の体系的比較では、Rustの逐次性能はFortranの**約1.2%遅い**、C++より**約5.6%速い**。40スレッド並列（rayon vs OpenMP）ではFortranの約16%遅く、C++の約19%遅い。この差はrayonの`nowait`相当機能の欠如が主因であり、embarrassingly parallelなワークロード（バッチ断面積計算等）では差はほぼ消失する。

### GPU高速化の実績と見積もり

| 計算ステージ | 手法 | 見込み高速化 | 根拠 |
|-------------|------|------------|------|
| 光学模型（バッチNumerov） | GPUアンサンブルODE | **50〜500倍** | DiffEqGPU.jlベンチマーク |
| HF合算 | バッチ線形代数 | **5〜20倍** | HPRMAT実績 |
| R-matrix線形ソルバ | GPU LU/混合精度 | **9〜18倍** | HPRMAT（RTX 3090） |
| χ²フィッティング | GPU LM | **42倍** | Gpufit（GTX 1080） |
| MACS積分 | GPU map-reduce | **10〜100倍** | 構造的見積もり |
| 全核図バッチ計算 | 全ステージ統合 | **100〜1000倍** | エンドツーエンド推定 |

TALYSの1核種あたり数分〜1時間の計算が、GPUバッチ処理により**数秒〜数十秒に短縮**される可能性がある。全核図計算（数千核種）では、現在のHPC数日規模の計算がデスクトップGPU1枚で**数時間に圧縮**される見込み。

### 混合精度戦略の重要性

コンシューマGPU（RTX 3090/4090）のFP64:FP32スループット比は**64:1**であり、HPRMATが実証したFP32 LU分解＋反復精緻化によるFP64精度回復は核物理計算全般に適用可能。断面積の相対精度10⁻⁵で十分な大半のHF計算では混合精度が有効。狭い共鳴（Γ ≲ 1 eV、条件数κ > 10⁸）近傍ではフルFP64が必要で、パラメータ領域に応じた自動精度切替機構を設計すべき。

---

## 実装ロードマップと推奨技術スタック

### フェーズ1：基盤（6ヶ月）

`nucrust-core`（型・単位系）、`nucrust-data`（RIPL-3/REACLIBパーサー）、`nucrust-special`（Coulomb関数の純Rust実装）を最優先で開発。Coulomb関数はThompson-Barnett COULCC アルゴリズムのクリーンルーム実装とし、`wide`クレートによるSIMDバッチ化（`f64x4`で4組の(η,ρ)を同時評価）を最初から設計に組み込む。

### フェーズ2：計算エンジン（6ヶ月）

`nucrust-optical`（Numerov法、球形光学模型）、`nucrust-hf`（HF統計模型）、`nucrust-rmatrix`（Lane-Thomas形式）をCPUバックエンド（rayon + faer）で実装。TALYS/AZURE2との数値一致検証を並行実施。

### フェーズ3：GPU加速（6ヶ月）

`nucrust-gpu`でcudarc経由のCUDA Cカーネルを統合。バッチNumerov→HF合算→MACS積分のエンドツーエンドGPUパイプラインを構築。混合精度戦略を実装。

### フェーズ4：エコシステム統合（3ヶ月）

`nucrust-python`（PyO3 + rust-numpy）でpynucastro/NumPy連携を提供。`nucrust-astro`で恒星反応率・r-process/s-process計算インターフェースを公開。

### 推奨技術スタック総括

- **線形代数**: faer（密行列）+ faer::sparse（疎行列）
- **特殊関数**: 純Rust Coulomb関数（新規開発）+ puruspe（γ/Bessel）+ rgsl（フォールバック）
- **複素数**: num-complex
- **GPU**: cudarc（ホスト管理）+ CUDA Cカーネル（RTCコンパイル）
- **SIMD**: wide（安定版Rust）+ std::arch生intrinsics
- **CPU並列**: rayon（データ並列）+ crossbeam（カスタム同期）
- **分散計算**: rsmpi（MPI 3.1）
- **Python連携**: PyO3 + rust-numpy + maturin
- **Fortran FFI**: extern "C" + cc + bind(C)
- **データI/O**: serde（TOML/JSON/MessagePack）+ hdf5-metno
- **テスト**: Criterion.rs + proptest + approx
- **ライセンス**: MIT OR Apache-2.0（デュアル）

---

## まとめ：世界初のGPU対応HFエンジンという位置づけ

本プロジェクトは「GPU並列化されたHauser-Feshbachコードが存在しない」という核天体物理計算における根本的な空白を埋めるものである。Rustの型安全性とゼロコスト抽象化により、Fortranコードベースの40年分の技術的負債（グローバル変数、COMMONブロック、ECIS-06のブラックボックス化）を解消しつつ、GPUバッチ並列化で**2〜3桁の高速化**を実現できる見通しがある。faerの密行列性能がBLAS/LAPACK同等に達し、cudarcが安定したCUDA FFIを提供し、rayonがOpenMP同等のデータ並列を実現する2026年は、このプロジェクトを開始する最適なタイミングである。最大の技術的リスクはCoulomb関数の純Rust高精度実装とGPUカーネルのワープダイバージェンス管理であり、これらを最初のマイルストーンとして重点的に取り組むことを推奨する。