# nucrust 実装タスクリスト

> **基準文書:** `docs/design.md` v1.0.0 (2026-03-17)
> **作成日:** 2026-03-19

凡例: `[ ]` 未着手 / `[x]` 完了

---

## 0. プロジェクト基盤

- [x] **T-0.1** ワークスペース Cargo.toml 作成（10クレート定義、feature flag設計 §1.3）
- [x] **T-0.2** 各クレートの Cargo.toml 雛形作成（依存関係グラフ §2.1 に準拠）
- [x] **T-0.3** ディレクトリ構成の構築（§1.2: `crates/`, `kernels/`, `benches/`, `tests/reference_data/`, `data/`）
- [x] **T-0.4** CI/CD パイプライン（`.github/workflows/ci.yml` §13.4: clippy, fmt, test）
- [x] **T-0.5** テスト用サンプルデータ準備（`data/ripl3_sample/`, `data/reaclib_sample/`）
- [x] **T-0.6** mpmath 検証データ生成スクリプト作成（`scripts/generate_mpmath_reference.py`, dps=60, 22点）

---

## Phase 1: 基盤 (M1–M6)

### 1A. nucrust-core（§3）

#### 型システム（§3.1）

- [x] **T-1A.1** `Nuclide` 型（§3.1.1: Z=0–118, A=1–350, バリデーション付きコンストラクタ）
- [x] **T-1A.2** `Projectile` 列挙型（§3.1.2: n/p/d/t/³He/α/γ、質量・スピン・電荷メソッド）
- [x] **T-1A.3** `Channel` 型（§3.1.3: projectile + target + Q値）
- [x] **T-1A.4** `SpinParity` 型（§3.1.3: `two_j: i32`, `Parity` 列挙型, `triangle_condition()`)
- [x] **T-1A.5** `EnergyGrid` 型（§3.1.4: 対数/線形/任意グリッド、ソート・正値保証）
- [x] **T-1A.6** `CubicSpline` 補間器（§3.1.4: 自然スプライン、`evaluate()`, `evaluate_batch()`）
- [x] **T-1A.7** `TransmissionCoeffs` 型（§3.1.5: SoA レイアウト `data[l][j_index][e_index]`）
- [x] **T-1A.8** `CrossSection` 型（§3.1.5: total/elastic/reaction + 部分断面積 `Vec<PartialCrossSection>`）
- [x] **T-1A.9** `CollisionMatrix` 型（§3.1.5: U行列 SoA `Vec<Complex64>`）
- [x] **T-1A.10** `ReactionRate` 型（§3.1.5: temperatures, NA<σv>, MACS, S因子, SEF）

#### 単位系（§3.2）

- [x] **T-1A.11** `units` モジュール（CODATA 2022 物理定数: `HBAR_C`, `AMU_MEV`, `FINE_STRUCTURE`, `BOLTZMANN_MEV`, `AVOGADRO`）
- [x] **T-1A.12** 変換関数群（`sommerfeld_parameter()`, `wave_number()`, `reduced_mass()`, `cm_energy()`）

#### エラー・トレイト（§3.3–3.5）

- [x] **T-1A.13** `CoreError` 列挙型（§3.3: thiserror, 8 バリアント）
- [x] **T-1A.14** `ComputeBackend` トレイト（§3.4: `batch_numerov`, `hf_summation`, `rmatrix_solve`, `macs_integrate`）
- [x] **T-1A.15** `NldModelParams` / `GsfModelParams` 列挙型（§3.4: GPU転送用パラメータ表現）
- [x] **T-1A.16** `ToDeviceParams` トレイト（§3.4）
- [x] **T-1A.17** `CpuBackend` 構造体スタブ（§3.4: rayon `ThreadPool`）
- [x] **T-1A.18** 物理モデルトレイト（§3.5: `LevelDensity`, `GammaStrength`, `OpticalPotential`）
- [x] **T-1A.19** `Multipole` 列挙型（§3.5: E1, M1, E2, M2, E3）

### 1B. nucrust-special（§4）

#### Coulomb 波動関数

- [x] **T-1B.1** 公開 API: `coulomb_wave()`, `CoulombResult`, `ScalingMode`（§4.2, §4.3.8）
- [x] **T-1B.2** 修正 Lentz 法 `continued_fraction_lentz<T>()`（§4.3.2: ε=1e-15, max_iter=20,000）
- [x] **T-1B.3** 連分数 CF1: $f_l = F'_l/F_l$（§4.3.2: $R_l, S_l, T_l$ 補助量、DLMF §33.8.1）
- [x] **T-1B.4** 連分数 CF2: $p+iq = H^{+\prime}_l/H^+_l$（§4.3.3: 複素連分数、DLMF §33.8.2）
- [x] **T-1B.5** Wronskian 結合（Steed 法）（§4.3.4: CF1+CF2 → F, G, F', G'）
- [x] **T-1B.6** 計算領域判別ロジック（§4.3.1: 振動領域/禁止領域/小ρ判別、Numerov逆積分）
- [x] **T-1B.7** 三項漸化式（§4.3.5: F下降/G上昇、DLMF §33.4）
- [x] **T-1B.8** べき級数展開（§4.3.6: $_1F_1$ 級数、$C_l(\eta)$ Gamow 因子）
- [x] **T-1B.9** 複素対数ガンマ関数 `complex_log_gamma()`（Spouge近似 a=15, ~13桁精度）
- [x] **T-1B.10** Coulomb 位相差 $\sigma_l$（§4.3.7: `Im[ln Γ(l+1+iη)]`）
- [x] **T-1B.11** 指数スケーリング（§4.3.8: Scaled/Unscaled モード — 型定義のみ、Scaled未実装）
- [x] **T-1B.12** バッチ API `coulomb_wave_batch()`（§4.2）

#### SIMD バッチ化

- [x] **T-1B.13** `wide` クレート `f64x4` による SIMD バッチ化（§4.4: f64x4ネイティブCF1/CF2実装、Steed法SIMD化、SmallRhoはスカラーフォールバック）

#### 検証テスト

- [x] **T-1B.14** Tier 1 テスト（§4.5: η=0 ベッセル極限、複数ρ・複数l）
- [x] **T-1B.15** Tier 2 テスト（§4.5: 禁止領域 η=10,ρ=5、Numerov逆積分）
- [x] **T-1B.16** Tier 3 テスト（§4.5: 高 l=10、η=5,ρ=10）
- [x] **T-1B.17** 自己整合性検証（§4.5: Wronskian=1, 交差積, σ_l(η=0)=0）
- [x] **T-1B.18** mpmath 50桁精度との比較テスト（ACC-01: |F|,|G| Tier1 < 1e-10, Tier2 < 1e-2, 交差積検証）

### 1C. nucrust-data（§5）

#### RIPL-3 パーサー

- [x] **T-1C.1** Fortran 固定幅フィールドパーサーユーティリティ（§5.2.2: `fixed_field_i32()`, `fixed_field_f64()`, `fixed_field_str()`）
- [x] **T-1C.2** 離散準位パーサー `parse_discrete_levels()`（§5.2.2: ヘッダ/準位/γ線行 FORMAT、欠損値処理）
- [x] **T-1C.3** `IsotopeData`, `DiscreteLevel`, `GammaTransition`, `MassEntry` 型定義（§5.2.2）
- [x] **T-1C.4** 光学模型パラメータパーサー `parse_omp_database()`（§5.2.3: `om-parameter-u.dat`）
- [x] **T-1C.5** 質量テーブルパーサー `parse_mass_table()`（§5.2.1）
- [x] **T-1C.6** レベル密度パラメータパーサー `parse_level_density_params()`, `parse_hfb_density_table()`（§5.2.1）
- [x] **T-1C.7** γ線強度パーサー `parse_gdr_params()`, `parse_gsf_table()`（§5.2.1）
- [x] **T-1C.8** 共鳴パラメータパーサー `parse_resonances()`（§5.2.1）
- [x] **T-1C.9** 核分裂障壁パーサー `parse_fission_barriers()`（§5.2.1）
- [x] **T-1C.10** 殻補正パーサー `parse_shell_corrections()`（§5.2.1）

#### REACLIB パーサー

- [x] **T-1C.11** REACLIB R1 パーサー `parse_reaclib()`（§5.3: 74文字固定幅、Chapter 1–11）
- [x] **T-1C.12** `ReaclibEntry` 型 + `evaluate(t9)` メソッド（§5.3: 7パラメータ公式）
- [x] **T-1C.13** REACLIB 7パラメータフィッティング `fit_reaclib_params()`（§5.4: faer QR 分解）

#### TOML 設定・HDF5

- [x] **T-1C.14** TOML 設定スキーマ（§5.6: `JobConfig` + serde Deserialize）
- [ ] **T-1C.15** HDF5 入出力（§5.5: feature `hdf5_io` コード実装済み、HDF5ライブラリ依存のビルド・テスト未実施）

---

## Phase 2: CPU 計算エンジン (M7–M12)

### 2A. nucrust-optical（§6）

#### 球形光学模型（Phase 1 priority）

- [x] **T-2A.1** `woods_saxon()`, `woods_saxon_deriv()`, `spin_orbit_factor()` ヘルパー（§6.2）
- [x] **T-2A.2** `KoningDelaroche` 構造体 + `OpticalPotential` トレイト実装（§6.2, §6.4）
- [x] **T-2A.3** `McFaddenSatchler` α粒子 OMP（§6.4）
- [x] **T-2A.4** `Avrigeanu2014` α粒子 OMP（§6.4）
- [x] **T-2A.5** `CustomOmp` ユーザー定義パラメータ OMP（§6.4）

#### Numerov 積分

- [x] **T-2A.6** `NumerovConfig` 設定型（§6.3.1: step_size, r_min, convergence_tl, max_l）
- [x] **T-2A.7** Fox-Goodwin 比変数法 `numerov_integrate()`（§6.3.1: 比変数 $R_n$ 漸化式）
- [x] **T-2A.8** S行列抽出（§6.3.2: 対数微分 $L_l$ → $S_l$, Coulomb 関数連携）
- [x] **T-2A.9** `compute_transmission_coeffs()`（§6.3.3: l-s 分離, l_max 収束判定）

#### 結合チャンネル（Phase 2 拡張）

- [x] **T-2A.10** 結合チャンネル Schrödinger 方程式の行列形式（§6.5.1: `CcChannel`, `CoupledChannelSystem`）
- [x] **T-2A.11** Johnson 対数微分法（§6.5.2: `deformation.rs` 結合ポテンシャル行列）
- [x] **T-2A.12** 変形パラメータ → 結合行列要素変換（§6.5.3: `DeformedKoningDelaroche`, Wigner記号）

#### 検証テスト

- [ ] **T-2A.13** TALYS 透過係数との比較（ACC-02: 相対誤差 < 10⁻⁶, ゴールデンファイル `fe56_ng_transmission.dat`）
- [x] **T-2A.14** proptest: $T_{lj} \in [0, 1]$, 有限性、ランダム (Z,A,E) で検証

### 2B. nucrust-hf（§7）

#### Hauser-Feshbach 中核

- [x] **T-2B.1** `HfConfig` 型（§7.1.2: j_max, max_particle_stages, max_gamma_steps, WfcModel）
- [x] **T-2B.2** Jπ 合算ループ（§7.1.3: 入射/出射チャンネル透過係数、三角条件・パリティ選択則）
- [x] **T-2B.3** 離散準位と連続準位の接続処理（§7.1.4: DiscreteLevels型、E_complete以下は離散準位、以上はNLD連続モデル、γ透過係数の離散/連続接続実装済み）
- [x] **T-2B.4** 連続準位積分: $\int T_{lj}(E-U) \cdot \rho(U) dU$（§7.1.3: 出射粒子チャンネルの連続準位積分実装、T_{lj}補間＋NLD畳み込み、離散準位との接続含む）
- [x] **T-2B.5** γ線チャンネル透過係数（§7.1.3: 多極子合算、$f_{XL}(E_γ) \cdot E_γ^{2L+1} \cdot \rho$）

#### 多粒子放出カスケード

- [x] **T-2B.6** 明示的スタック方式カスケード計算 `cascade_calculation()`（§7.2: HF分岐比、粒子放出→子核プッシュ、γカスケード）
- [x] **T-2B.7** γカスケード計算 `gamma_cascade()`（§7.2: 統計的モデル、E1/M1/E2多極子競合、NLDベース遷移先選択）

#### 幅揺らぎ補正 (WFC)

- [x] **T-2B.8** Moldauer WFC 1次元積分（§7.3.1: $W_{ab}$ 補正因子、弾性散乱増強）
- [x] **T-2B.9** Kawano-Talou $\nu_a$ パラメータ化（§7.3.2: GOE 最良フィット）
- [x] **T-2B.10** GOE 三重積分（§7.3.4: VWZ 公式、`goe_wfc()` Gauss-Laguerre/Legendre 求積）

#### NLD/GSF モデル

- [x] **T-2B.11** NLD 実装: CT, BSFG, GilbertCameron, Ignatyuk, **HfbTableInterp** 全モデル実装済み（§7.4）
- [x] **T-2B.12** GSF 実装: SLO, EGLO, **QrpaTableInterp** 全モデル実装済み（§7.4）
- [x] **T-2B.13** `LevelDensity` + `GammaStrength` トレイト実装（各モデル）

#### 検証テスト

- [ ] **T-2B.14** ⁵⁶Fe(n,γ) HF 断面積の TALYS 比較（ACC-03: 相対誤差 < 10⁻⁶, ゴールデンファイル `fe56_ng_cross_section.dat`）
- [x] **T-2B.15** proptest: 断面積 ≥ 0、ランダムエネルギーで検証

### 2C. nucrust-rmatrix（§8）

#### R-matrix 計算

- [x] **T-2C.1** `RMatrixParams`, `RMatrixChannel`, `RMatrixLevel`, `BoundaryCondition` 型（§8.1.2）
- [x] **T-2C.2** R行列構築 + $(1-RL^0)^{-1}$ LU 分解（§8.1.1: faer 使用）
- [x] **T-2C.3** 衝突行列 $U_{cc'}$ 構成（§8.1.1: 透過因子 $P_c$, Coulomb 位相因子 $\Omega_c$）
- [x] **T-2C.4** `rmatrix_cross_section()` → `RMatrixResult`（§8.1.3）

#### Brune 変換

- [x] **T-2C.5** 非線形固有値問題ソルバー（§8.2.1: Newton 法、収束閾値 1e-12）
- [x] **T-2C.6** `standard_to_brune()` / `brune_to_standard()` 変換関数（§8.2.2）
- [x] **T-2C.7** 準位シフト関数 $S_c(E)$ + $dS_c/dE$ の計算（§8.2.3）

#### χ² フィッティング

- [x] **T-2C.8** Levenberg-Marquardt 法 `levenberg_marquardt()`（§8.3.1: 有限差分 Jacobian）
- [x] **T-2C.9** Affine-invariant MCMC `mcmc_sample()`（§8.3.2: Goodman-Weare, ストリーミング HDF5 出力）

#### 検証テスト

- [ ] **T-2C.10** ⁷Be(p,γ)⁸B R-matrix の AZURE2 比較（ACC-04: 相対誤差 < 10⁻⁶, `be7_pg_rmatrix.dat`）

### 2D. CpuBackend 統合

- [x] **T-2D.1** `CpuBackend` の `ComputeBackend` トレイト完全実装（hf_summation全NLD/GSFモデル対応、rmatrix_solve実計算、macs_integrate実装済み）
- [x] **T-2D.2** エンドツーエンドパイプライン統合テスト（透過係数 → HF → 断面積、3テスト）

---

## Phase 3: GPU 加速 (M13–M18)

### 3A. nucrust-gpu（§10）

#### GPU バックエンド基盤

- [x] **T-3A.1** `GpuBackend` 初期化（cudarc v0.16 `CudaContext` + `CudaStream`, NVRTC コンパイル、feature "cuda"）
- [x] **T-3A.2** `GpuModuleCache` カーネルキャッシュ骨格（§10.1）
- [x] **T-3A.3** GPU メモリプール `GpuMemoryPool`（§10.7: サブアロケーション、VRAM予算制限）
- [x] **T-3A.4** 混合精度自動選択 `PrecisionStrategy`（CC検出→Full64/Mixed32Refine/Pure32 自動選択）

#### CUDA C カーネル

- [x] **T-3A.5** `kernels/numerov.cu` — バッチ Numerov カーネル（Fox-Goodwin 比変数法、BLOCK_SIZE=256、NVIDIA L4 テスト済み）
- [x] **T-3A.6** ワープダイバージェンス緩和: 2段階ビンニング `bin_numerov_tasks()`（l→energy ソート、unsort復元）
- [x] **T-3A.7** `kernels/hf_summation.cu` — HF 合算カーネル（J-pi ループ per thread、CT NLD + SLO GSF in-kernel）
- [x] **T-3A.8** `kernels/rmatrix_solve.cu` — R-matrix batched in-thread Gauss-Jordan（N≤10チャンネル、cuSOLVER不使用）
- [x] **T-3A.9** `kernels/macs_integral.cu` — MACS Gauss-Laguerre 8点積分カーネル

#### GPU Coulomb 関数

- [x] **T-3A.10** Phase 1: テーブル方式（§10.6: CPU事前計算、GPU global memoryアップロード、`build_coulomb_table()`）
- [x] **T-3A.11** オンザフライ `__device__` 関数（CF1+CF2+Steed CUDA実装、Bessel極限テスト済み）
- [x] **T-3A.12** ハイブリッド方式（テーブル/オンザフライ自動分類、ワープ分離ソート）

#### GPU パイプライン

- [x] **T-3A.13** `batch_pipeline()`（§10.8: Numerov→HF→MACS 連続 GPU 実行）
- [x] **T-3A.14** `GpuBackend` の `ComputeBackend` トレイト実装（hf_summation + macs_integrate 委譲）
- [x] **T-3A.15** マルチストリーム非同期実行（`MultiStreamExecutor`: N ストリーム管理、同期、fork）

#### hipify 互換設計

- [x] **T-3A.16** CUDA C カーネルの hipify-clang 互換性確認（動的並列処理なし、cuComplex不使用、extern "C" __global__、テクスチャなし）

### 3B. ベンチマーク検証

- [x] **T-3B.1** PERF-B1: ⁵⁶Fe(n,γ) 透過係数 6,200タスク GPU **0.408s** (< 1秒)
- [ ] **T-3B.2** PERF-B2: ⁵⁶Fe(n,γ) HF 断面積 GPU 0.080s warm（目標0.05s、NVRTC再コンパイル含む — カーネルキャッシュで改善余地）
- [x] **T-3B.3** PERF-B3: ²³⁸U(n,γ) HF 断面積 200E×l=40 GPU **0.618s** (< 5秒)
- [x] **T-3B.4** PERF-B4: ⁷Be(p,γ)⁸B R-matrix 1000E×3lvl GPU **0.210s** (< 5秒)
- [x] **T-3B.5** ACC-06: GPU FP64 TC値が [0,1] 範囲で有限であることを確認

---

## Phase 4: エコシステム (M19–M24)

### 4A. nucrust-astro（§9）

- [x] **T-4A.1** MACS 計算 `compute_macs()`（§9.1: Gauss-Laguerre 20点求積、CubicSpline 補間）
- [x] **T-4A.2** 天体物理反応率 `compute_reaction_rate()`（§9.2: $N_A\langle\sigma v\rangle(T)$、FCZ定数）
- [x] **T-4A.3** S因子 `compute_s_factor()`（§9.3: $S(E) = \sigma E \exp(2\pi\eta)$）
- [x] **T-4A.4** 恒星増強因子 `compute_sef()`（§9.4: 分配関数＋励起状態寄与）

### 4B. nucrust-python（§11）

- [x] **T-4B.1** PyO3 モジュール定義 `#[pymodule]`（§11.1: PyNuclide, PyReactionRate, PyCrossSection）
- [x] **T-4B.2** NumPy 連携（§11.2: `numpy` crate `PyArray1::from_slice_bound`）
- [x] **T-4B.3** Python 関数ラッパー（§11.1: `calc_transmission_coeffs`, `calc_hf_cross_section`, `calc_macs`, `fit_reaclib`）
- [x] **T-4B.4** maturin ビルド設定（`pyproject.toml`）
- [ ] **T-4B.5** pynucastro 連携テスト（§11.3: REACLIB 出力→pynucastro 読み込み）

### 4C. nucrust CLI（§12）

- [x] **T-4C.1** clap サブコマンド定義（§12.2: `calc`, `batch`, `fit`, `info`, `export`）
- [ ] **T-4C.2** TOML 設定ファイル連携（§12.2: calc のみ実装、batch/fit/info/export は stub）
- [ ] **T-4C.3** 出力フォーマット切り替え（§12.3: JSON/TSV 実装済み、HDF5/REACLIB 未実装）
- [x] **T-4C.4** バックエンド選択（§12.2: `--backend cpu|gpu` フラグ定義済み）

### 4D. 大規模検証

- [ ] **T-4D.1** PERF-B5: ¹²C(α,γ)¹⁶O MCMC 10,000サンプル GPU < 30分
- [ ] **T-4D.2** PERF-B6: 全核図バッチ 10,000核種 GPU < 2時間
- [ ] **T-4D.3** ACC-05: 反応率 vs 先行計算 差 < 5%

### 4E. ドキュメント・リリース

- [ ] **T-4E.1** rustdoc 100%（public API）
- [ ] **T-4E.2** mdbook ユーザーガイド（`book/`）
- [ ] **T-4E.3** CHANGELOG.md 作成
- [ ] **T-4E.4** REACLIB 出力が pynucastro で読み込み可能であることの受け入れテスト
