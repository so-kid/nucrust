# nucrust — Software Requirements Specification

> **⚠️ ステータス: 凍結（歴史的文書）** — 本文書は実装開始前の調査・設計時スナップショットであり、以後更新されません。現在の仕様は `book/`（mdBook ユーザーガイド）と rustdoc、実装状況は `CHANGELOG.md` / `task.md` を参照してください。（2026-06-12 注記）

> **Rust製原子核反応率計算フレームワーク**
> GPU加速 Hauser-Feshbach / R-matrix 計算エンジン

| 項目 | 値 |
|------|-----|
| バージョン | 0.2.1-draft |
| 日付 | 2026-03-17 |
| ライセンス | MIT OR Apache-2.0 |
| ステータス | Draft (Rev.2) |

### 改訂履歴

| バージョン | 日付 | 概要 |
|-----------|------|------|
| 0.1.0 | 2026-03-17 | 初版 |
| 0.2.0 | 2026-03-17 | 性能目標のベンチマーク問題定義化、R-matrix GPU戦略追加、Coulomb関数検証階層見直し、多粒子カスケード明確化、混合精度適用層の明示、不確かさ伝搬スコープ明確化 |
| 0.2.1 | 2026-03-17 | CUDA専一方針の確定、AWS GPU調達戦略の明記、CubeCL評価の追記、GPU CI戦略の具体化、hipify考慮設計の追加 |

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [機能要件](#2-機能要件)
3. [非機能要件](#3-非機能要件)
4. [アーキテクチャ設計](#4-アーキテクチャ設計)
5. [リスクと制約](#5-リスクと制約)
6. [実装ロードマップ](#6-実装ロードマップ)
7. [受け入れ基準](#7-受け入れ基準)
- [付録A: 用語定義](#付録a-用語定義)
- [付録B: 検証参照実装](#付録b-検証参照実装)
- [付録C: ベンチマーク問題定義](#付録c-ベンチマーク問題定義)

---

## 1. プロジェクト概要

### 1.1 背景と動機

宇宙元素合成のシミュレーションにおいて、原子核反応率の計算は最も計算コストの高いステージである。現在の標準コード群（TALYS, AZURE2, SMARAGD, CoH3, EMPIRE等）はいずれもFortran/C++によるシリアル実行であり、GPU並列化されたHauser-Feshbachコードは世界に存在しない。本プロジェクト「nucrust」は、Rust言語によるGPU加速反応率計算エンジンをコミュニティ向けOSSライブラリとして構築する。

### 1.2 スコープ

本フレームワークは「反応率計算エンジン」層に特化する。

**スコープ内:**

- Hauser-Feshbach統計模型による複合核反応断面積計算
- R-matrix理論による共鳴反応断面積計算・フィッティング
- 光学模型ポテンシャルからの透過係数計算
- 天体物理反応率（MACS、SEF、S因子）の導出
- RIPL-3 / REACLIB / STARLIBデータ形式の読み書き

**スコープ外:**

- 反応ネットワークソルバー（SkyNet/WinNet相当）
- 流体力学コードとの結合
- ab initio核構造計算
- 入力パラメータ不確かさの反応率への伝搬（感度解析・Monte Carlo伝搬）

> **v0.2.0追記:** 不確かさ伝搬はPhase 4以降の将来機能として記録する。STARLIB形式の不確かさ分位点読み込みはFR-IO-02で対応するが、OMP係数やNLDパラメータ等の入力不確かさを反応率に伝搬する計算機能は現フェーズのスコープに含まない。

### 1.3 設計目標の優先順位

| 順位 | 目標 | 説明 |
|------|------|------|
| 1 | 計算性能 | GPU/SIMD/並列化による既存コード比 50〜500倍の高速化 |
| 2 | 正確性 | 参照実装（付録B参照）との数値一致 |
| 3 | 拡張性 | トレイトベースの物理モデルプラグイン機構 |
| 4 | 相互運用性 | Pythonバインディング、Fortran FFI（検証目的）、C ABI公開 |

> **v0.2.0追記:** Fortran FFIの目的を「検証目的」と明確化。既存TALYS/AZURE2サブルーチンの呼び出しは参照値生成のためのテストハーネスとして使用し、本番運用での利用は想定しない。

### 1.4 想定ユーザー

- 核天体物理学の研究者（反応率計算、元素合成シミュレーション）
- 核データ評価者（TENDL, JENDL, ENDFライブラリ生成）
- 実験核物理学者（R-matrixフィット、S因子外挿）
- フレームワーク開発者（pynucastro, MESA等との統合）

---

## 2. 機能要件

### 2.1 光学模型・透過係数計算 (FR-OMP)

#### FR-OMP-01: 球形光学模型

Woods-Saxon型ポテンシャル（実部体積項、虚部体積/表面項、スピン軌道項、Coulomb項）を実装し、Numerov法によるSchrödinger方程式の数値積分から透過係数 T_l(E) を算出する。

- Koning-Delaroche (2003) 大域的核子OMPパラメータをデフォルト提供
- α粒子OMP: McFadden-Satchler, Avrigeanu等を選択可能
- カスタムOMPパラメータのユーザー定義サポート
- 部分波 l_max の自動決定（収束判定付き）

#### FR-OMP-02: 結合チャンネル光学模型

変形核に対する結合チャンネル方程式の数値解法。回転対称核の2+, 4+状態への結合を含む。Phase 2以降で実装。

#### FR-OMP-03: Coulomb波動関数

正則・COULOMB波動関数 F_l(η, ρ), G_l(η, ρ) およびその導関数を純Rustで実装する。Thompson-Barnett COULCCアルゴリズムのクリーンルーム実装とし、SIMDバッチ計算を設計に組み込む。

> **v0.2.0改訂:** 検証参照をGSL単独から階層化に変更。GSLの `gsl_coulomb_wave_*` は η >> 1 かつ ρ << η の準古典的領域で精度低下が知られているため、GSL単独を検証基準としない。

**検証階層（下位が優先）:**

1. **第1層:** mpmath任意精度計算（50桁以上）との相対誤差 < 10⁻¹²
2. **第2層:** オリジナルThompson-Barnett Fortran実装（COULCC.f）との一致
3. **第3層:** GSL `gsl_coulomb_wave_F_array` との比較（GSLが高精度な領域のみ）

**カバレッジ:** η ∈ [0.01, 100], ρ ∈ [0.001, 1000], l ∈ [0, 50] の網羅的グリッド

### 2.2 Hauser-Feshbach統計模型 (FR-HF)

#### FR-HF-01: 複合核断面積

Hauser-Feshbach公式による複合核反応断面積の計算。入射チャンネル（n, p, d, t, ³He, α, γ）と出射チャンネルの透過係数およびレベル密度を用いたJπ合算。

- 幅揺らぎ補正（WFC; Moldauer/GOE形式）
- 離散準位と連続準位の統合的取り扱い

#### FR-HF-01a: 多粒子放出カスケード

> **v0.2.0新規:** 「最大10段階」の定義を明確化

複数粒子放出の逐次計算を以下の通り定義する:

- **粒子放出段階:** 最大10段階まで。「1段階」は粒子（n, p, d, t, ³He, α）の放出1回を指す
- **γ線カスケード:** 各粒子放出段階の残留核からのγ崩壊を含む。γカスケードは最大ステップ数を別途指定（デフォルト: 30）
- **計算打ち切り:** 残留核の励起エネルギーが粒子分離エネルギー以下になった時点でγ崩壊のみに移行

重核（A > 200）の高励起状態からの全γカスケード計算は、Jπ状態空間の爆発によりPERF-B2を超過する可能性がある。この場合の挙動はベンチマーク問題B3（付録C）で別途評価する。

#### FR-HF-02: 核レベル密度モデル

プラグインとして提供:

- 定温度（CT）/ 逆移動Fermiガス（BSFG）/ Gilbert-Cameron複合
- Ignatyuk殻補正付きFermiガス
- Skyrme HFB組合せ法（テーブル補間）

#### FR-HF-03: γ線強度関数

プラグインとして提供:

- SLO / EGLO (Kopecky-Uhl) / 微視的QRPAテーブル / M1強度関数

### 2.3 R-matrix理論 (FR-RM)

#### FR-RM-01: Lane-Thomas R-matrix

多準位・多チャンネルR-matrix計算。衝突行列、全断面積、微分断面積、放射捕獲断面積の算出。

- Brune代替パラメータ化のサポート
- バックグラウンド極の取り扱い
- 直接捕獲成分との干渉

#### FR-RM-02: χ²フィッティング

Levenberg-Marquardt法およびMCMC（Metropolis-Hastings / Affine-invariant）によるR-matrixパラメータ最適化。

> **v0.2.0追記:** PERF-B4の前提条件を明確化。ベンチマーク問題B4（付録C）で実験データ点数、パラメータ数、反復回数を定義。

### 2.4 天体物理反応率 (FR-ASTRO)

#### FR-ASTRO-01: Maxwell平均断面積・反応率

Gauss-Laguerre求積法による N_A⟨σv⟩ の計算。0.001〜10 GK、20〜60温度グリッド点。

- 事前計算済み断面積テーブルの補間による計算をデフォルトとする（HFフル再計算はオプション）

#### FR-ASTRO-02〜04

S因子、恒星増強因子（SEF）、REACLIB 7パラメータ自動フィット出力。

### 2.5 データ入出力 (FR-IO)

- **FR-IO-01:** RIPL-3パーサー（8セクション: 質量、離散準位、共鳴、OMP、NLD、γSF、核分裂、殻補正）
- **FR-IO-02:** REACLIB R1形式 / STARLIB表形式（不確かさ分位点含む）の読み書き
- **FR-IO-03:** HDF5入出力（hdf5-metno、圧縮対応）
- **FR-IO-04:** TOML設定ファイル（serde型安全）

### 2.6 CLI (FR-CLI)

> **v0.2.0新規追加**

nucrustクレートにCLIを提供する。

- サブコマンド構成: `nucrust calc`（1核種）, `nucrust batch`（全核図）, `nucrust fit`（R-matrixフィット）
- 入力: TOML設定ファイル指定 またはコマンドラインパラメータ
- 出力フォーマット: `--format json|hdf5|reaclib|table`
- バックエンド選択: `--backend cpu|gpu`

---

## 3. 非機能要件

### 3.1 性能要件——ベンチマーク問題による定義

> **v0.2.0大幅改訂:** 「典型的な中重核」という曖昧な前提を廃止し、付録Cの代表的ベンチマーク問題に対する具体的目標値として再定義。

| ID | 要件（ベンチマーク問題参照） | 目標値 |
|----|---------------------------|--------|
| PERF-B1 | B1: ⁵⁶Fe(n,γ) 透過係数計算 (1,000E点 × 30部分波, 球形OMP) | GPU: 30,000積分 < 1秒 (A100/H100)、CPU: < 5秒 (8コア) |
| PERF-B2 | B2: ⁵⁶Fe(n,γ) HF断面積 (球形, 粒子放出2段階+γカスケード) | CPU: < 1秒、GPU: < 0.05秒 |
| PERF-B3 | B3: ²³⁸U(n,γ) HF断面積 (変形核, 多段階放出+全γカスケード) | CPU: < 60秒、GPU: < 5秒 ※変形核は球形核より1–2桁遅いことを許容 |
| PERF-B4 | B4: ⁷Be(p,γ)⁸B R-matrixフィット (9準位, 3粒子対, 16パラメータ, 500データ点, LM法 500反復) | CPU: < 30秒、GPU: < 5秒 |
| PERF-B5 | B5: ¹²C(α,γ)¹⁶O R-matrix MCMC (16パラメータ, 1,000データ点, 10,000 MCMCサンプル) | GPU: < 30分 (A100/H100) ※1サンプルあたり < 0.2秒 |
| PERF-B6 | B6: 全核図バッチ計算 (10,000核種, 球形OMP, HF+MACS) | GPU 1基: < 2時間、CPU 32コア: < 24時間 |
| PERF-M1 | メモリ使用量 | CPU: < 4GB、GPU VRAM: < 8GB |

> **v0.2.1変更:** GPU参照ハードウェアをRTX 4090からA100/H100に変更。AWS EC2（g5/p4d/p5）での検証を前提とする。コンシューマGPU（RTX 4090等）での性能はFP64スループット比に基づき推定値として記載する。

各ベンチマーク問題の詳細定義は[付録C](#付録c-ベンチマーク問題定義)を参照。

### 3.2 精度要件

| ID | 要件 | 目標値 |
|----|------|--------|
| ACC-01 | Coulomb波動関数 | mpmath (50桁) との相対誤差 < 10⁻¹² |
| ACC-02 | 透過係数（B1問題） | TALYSとの相対誤差 < 10⁻⁶ |
| ACC-03 | HF断面積（B2問題） | TALYSとの相対誤差 < 10⁻⁶ |
| ACC-04 | R-matrix断面積（B4問題） | AZURE2との相対誤差 < 10⁻⁶ |
| ACC-05 | 反応率 (MACS) | 先行計算との差 < 5% |
| ACC-06 | GPU混合精度計算 | FP64参照値との相対誤差 < 10⁻⁵ |

### 3.3 拡張性要件

#### NFR-EXT-01: トレイトベースの物理モデル

NLD、γSF、OMPの各カテゴリにRustトレイトを定義し、ユーザーが新モデルを実装・登録可能とする。

```rust
pub trait LevelDensity: Send + Sync {
    fn rho(&self, nuclide: &Nuclide, excitation: f64, spin: f64, parity: Parity) -> f64;
}
```

#### NFR-EXT-02: バックエンド抽象

`ComputeBackend`トレイトでCPU/GPU実装をコンパイル時feature flagで切り替え。

```rust
pub trait ComputeBackend {
    fn batch_numerov(params: &NumerovParams) -> TransmissionCoeffs;
    fn hf_summation(tc: &TransmissionCoeffs, ld: &LevelDensity) -> CrossSection;
    fn rmatrix_solve(rmat: &RMatrix, energies: &[f64]) -> CollisionMatrix;
}
```

### 3.4 相互運用性要件

- **Python:** PyO3 + rust-numpy によるNumPyゼロコピー連携
- **C ABI:** cdylib + cbindgenによるC互換ヘッダー自動生成
- **Fortran FFI:** 検証目的専用のテストハーネスとして提供

### 3.5 品質要件

- テストカバレッジ: line coverage 80%以上 + 解析解一致テストの物理ドメインカバー率追跡
- 回帰ベンチマーク: Criterion.rs + IaiによるCI性能回帰検出
- ファズテスト: cargo-fuzzによるパーサーモジュールのファズ
- 性質ベーステスト: proptestで断面積≥0、ユニタリ性、詳細釣合検証
- ドキュメント: rustdoc 100% (public API) + mdbookユーザーガイド

---

## 4. アーキテクチャ設計

### 4.1 クレート構成

| クレート | 役割 | 主要依存 |
|---------|------|---------|
| `nucrust-core` | 型定義、単位系、エラー | thiserror, serde |
| `nucrust-special` | Coulomb/Bessel関数 | num-complex, wide |
| `nucrust-data` | RIPL-3/REACLIBパーサー | serde, nom |
| `nucrust-optical` | 光学模型・透過係数 | faer, rayon |
| `nucrust-hf` | Hauser-Feshbach統計模型 | faer, rayon |
| `nucrust-rmatrix` | R-matrix理論 | faer, rayon |
| `nucrust-astro` | 天体物理反応率 | nucrust-hf, nucrust-rmatrix |
| `nucrust-gpu` | GPUバックエンド | cudarc |
| `nucrust-python` | Pythonバインディング | pyo3, rust-numpy |
| `nucrust` | 集約クレート・CLI | clap, 上記すべて |

> 依存クレートの具体バージョンはSRSに記載せず Cargo.lock で管理する。

### 4.2 GPU並列化戦略

#### 4.2.1 光学模型・透過係数 (FR-OMP)

**バッチNumerov:** エネルギー点×部分波×核種を3次元GPUスレッドグリッドにマッピング。1積分あたり約16KBでL1/共有メモリに収容。

**ワープダイバージェンス緩和（2段階戦略）:**

1. パラメータ空間を予想ステップ数でビンニングし、同一ビン内の積分を同一ワープに割り当て
2. ビン内の最大ステップ数でカーネルを起動し、早期収束スレッドは結果書き込み後にidle
3. 目標ワープ効率: ビンニング後に √(max/min) < 2 となるビン幅を確保

#### 4.2.2 Hauser-Feshbach合算 (FR-HF)

Jπ合算はmap-reduce演算としてGPU化。外側Jπループ×内側チャンネル合算を2Dスレッドグリッドにマッピング。分母（全透過係数総和）はワープシャッフル+共有メモリリダクション。

#### 4.2.3 R-matrix計算 (FR-RM)

> **v0.2.0新規追加:** レビュー指摘#2への対応

R-matrix計算のGPU並列化は以下の3層で構成する:

| 層 | 並列化単位 | 典型並列度 | GPUカーネル |
|----|-----------|-----------|------------|
| A: エネルギー | 各E点の行列反転 | 500〜5,000 | batched zgetrf / カスタム |
| B: LMフィット | パラメータ微分 | 16〜30 | 層AのN_param回呼び出し |
| C: MCMC | ウォーカー尤度 | 32〜256 | 層AのN_walker回呼び出し |

**層A: エネルギー点並列** — チャンネル行列 (1 − RL₀)⁻¹ の計算は各エネルギー点で完全に独立。典型的なフィットでは500〜5,000エネルギー点があり、各点でN_ch×N_ch行列（典型3〜20）の複素LU分解が必要。batched cuBLASまたはカスタムカーネルで実行。

**層B: フィッティング層 (LM法)** — コスト関数評価（全エネルギー点での断面積計算 + χ²計算）を層AのGPUバッチで実行。Jacobianの各列（パラメータ微分）は独立に計算可能であり、N_param個のコスト関数評価を並列化。GPU上のパラメータ更新とHessian構築も含む。

**層C: MCMC層** — Affine-invariant MCMCの各ウォーカーの尤度評価を層AのGPUバッチにマッピング。N_walker個の尤度評価を一括実行し、ホスト側でウォーカー更新を行う。ホスト↔デバイス間転送はパラメータベクトル（~100 float64）のみで、転送ボトルネックは発生しない。

#### 4.2.4 MACS積分 (FR-ASTRO)

Gauss-Laguerre求積ノードと重みをGPUコンスタントメモリに格納。各スレッドが1つの（核種, 温度）組合せの重み付き和を計算。事前計算済み断面積テーブルの補間による。

#### 4.2.5 エンドツーエンドGPUパイプライン

光学模型→透過係数→HF合算→MACS積分の全ステージをGPU上で連続実行し、ホスト↔デバイス間データ転送を排除する。PERF-B6の達成に不可欠。

### 4.3 混合精度戦略

> **v0.2.0改訂:** 適用層を明示的に分離。レビュー指摘#5への対応。

コンシューマGPUのFP64:FP32スループット比が64:1であるため、計算ステージごとに最適な精度を選択する:

| 計算ステージ | 精度 | 根拠 |
|-------------|------|------|
| ポテンシャル行列評価 | FP32 | Woods-Saxonポテンシャル値は相対精度 10⁻⁷ で十分 |
| Numerov積分ステップ | **FP64必須** | 漸化式の逐次累積誤差がFP32では破綻的 |
| S行列構築 | **FP64必須** | ユニタリ性保存に高精度が必要 |
| R-matrix行列反転 | FP32+反復精練化 | HPRMAT方式: FP32 LU→反復精練化でFP64回復 |
| HF Jπ合算 | FP64 | 透過係数の桁違いの差による桁落ち防止 |
| MACS積分 | FP64 | 求積重みの累積精度保持 |
| 狭共鳴近傍 (Γ < 1 eV) | **FP64必須** | 条件数 κ > 10⁸ ではFP32+反復精練化も失敗 |

**自動精度切替:** 共鳴幅とエネルギーグリッド間隔の比から条件数を推定し、閾値超過時にフルFP64に切り替え。

> **注記（v0.2.1）:** AWS EC2のA100/H100はFP64:FP32比が1:2であり、コンシューマGPUの1:64とは大きく異なる。A100/H100環境ではFP64ネイティブ実行がコスト効率で優位となるケースがあり、バックエンドがハードウェア特性に応じて自動選択する機構を設計する。

### 4.4 技術スタック

| カテゴリ | 採用技術 | 備考 |
|---------|---------|------|
| 線形代数 | faer | 純Rust、BLAS同等性能 |
| 特殊関数 | 純Rust新規実装 + puruspe | Coulomb関数が最重要 |
| 複素数 | num-complex | 安定版標準 |
| GPUランタイム | cudarc | CUDA専一。NVIDIA CC 7.0+（Volta以降）を対象。開発時はCPUバックエンドで代替し、GPU検証はAWS EC2（g5/p4d/p5系）を都度調達する運用を前提 |
| GPUカーネル言語 | CUDA C | hipifyによるHIP変換を将来の移行パスとして設計時に考慮する。ただしPhase 4終了時点まではCUDA C単一実装を維持 |
| SIMD | wide + std::arch | 安定版Rust対応 |
| CPU並列 | rayon | ワークスティーリング |
| 分散計算 | rsmpi | MPI 3.1、Phase 4 |
| Python | PyO3 + rust-numpy + maturin | ゼロコピー |
| データ I/O | serde + hdf5-metno | TOML/JSON/HDF5 |
| テスト | Criterion + proptest + approx | 回帰+性質+精度 |
| CLI | clap | サブコマンド構成 |

### 4.5 GPU CI/CD戦略

> **v0.2.1新規追加**

| テスト種別 | 実行環境 | 頻度 | 対象 |
|-----------|---------|------|------|
| CPUユニット・統合テスト | GitHub Actions (標準runner) | 全PR・pushごと | 全クレート |
| GPU精度テスト (ACC-06) | GitHub Actions GPU runner (T4/L4) | 全PR・pushごと (Phase 3以降) | nucrust-gpu |
| GPUベンチマーク (PERF-B1〜B5) | AWS EC2 g5.xlarge (A10G) Spot | 週次 | nucrust-gpu |
| 全核図バッチ (PERF-B6) | AWS EC2 p4d.24xlarge (A100×8) Spot | 月次 / リリース前 | nucrust-gpu + nucrust-astro |
| 最高性能ベンチマーク | AWS EC2 p5.48xlarge (H100×8) | リリース前のみ | 全ベンチマーク |

---

## 5. リスクと制約

### 5.1 技術リスク

| 影響度 | リスク | 緩和策 | 備考 |
|-------|-------|--------|------|
| 高 | Coulomb関数の純Rust実装 | GSL FFI + COULCC.fフォールバック | 3層検証階層を確立。クリティカルパス |
| 高 | GPUワープダイバージェンス | 2段階ビンニング戦略 | ワープ効率目標を定量化 |
| 高 | 多粒子カスケードのJπ空間爆発 | γカスケード打切り + B3ベンチマークで評価 | 重核は球形核より1–2桁遅い |
| 中 | Rust GPUエコシステムの未成熟 | CUDA Cカーネルを維持し、RustネイティブカーネルへはCubeCLのHPC実績を見て判断 | CubeCLはROCm/CUDA両対応のRust GPU計算FWとして開発中だが、FP64精度制御・混合精度の精緻な管理への適合性は未検証（2026年3月時点） |
| 中 | GPU調達コストとCI環境の制約 | 開発はCPUバックエンドで継続し、GPU検証はAWS EC2 Spot（g5.xlarge等）を都度調達。CIのGPUテストはPhase 3以降に限定 | g5.xlarge (A10G, sm_86) は開発・単体検証、p4d.24xlarge (A100×8) はPERF-B6検証に使用 |
| 中 | cudarc + rsmpiのMPI+CUDA統合 | Phase 4で単一ノードmulti-GPUから開始 | 実績少ない組合せ |
| 低 | AMD ROCm非対応による貢献者層の制限 | カーネルコードは将来のhipify変換を意識した構造（アトミック演算・warp intrinsicsの局所化）を維持。ROCm正式対応はPhase 4以降のコミュニティ要望に応じて判断 | RX 7000系 (gfx1100) はROCm 6.1+で正式サポート済みであり技術的障壁は低い |
| 低 | std::simd未安定化 | wideクレートで代替 | nightly不要 |

### 5.2 制約条件

- **ライセンス:** MIT OR Apache-2.0 デュアル
- **Rust:** stable (MSRV 1.80+)、nightly不要
- **GPU:** NVIDIA CUDA (Compute Capability 7.0+) 専一。AMD ROCmはPhase 4以降評価
- **FP64必須:** 混合精度は許容、FP32のみは不可
- **GPU調達方針:** 開発環境としてNVIDIA GPUを必須としない。GPUカーネルの継続的検証にはAWS EC2 GPUインスタンスを都度調達する運用を標準とし、ローカルGPU環境の有無によって開発参加が制限されない設計を維持する

---

## 6. 実装ロードマップ

| フェーズ | 期間 | 成果物 | マイルストーン |
|---------|------|--------|--------------|
| Phase 1 基盤 | M1–M6 | nucrust-core, data, special | Coulomb関数が3層検証パス、RIPL-3パーサー完成 |
| Phase 2 エンジン | M7–M12 | nucrust-optical, hf, rmatrix | B1–B4ベンチマークでTALYS/AZURE2と数値一致 |
| Phase 3 GPU | M13–M18 | nucrust-gpu, ベンチマーク | PERF-B1 50x+高速化達成、PERF-B4 GPU加速確認 |
| Phase 4 エコシステム | M19–M24 | nucrust-astro, python, CLI, multi-GPU | crates.io/PyPI公開、B6バッチ目標達成、ROCm評価レポート |

> **v0.2.0変更:** Phase 4の期間をM19–M21からM19–M24に拡張。MPI+CUDA統合とROCm評価の工数を反映。

---

## 7. 受け入れ基準

### 7.1 機能受け入れ

- 付録Cのベンチマーク問題B1–B4でTALYS/AZURE2との数値一致を確認
- REACLIB形式出力がpynucastroで読み込み可能
- 解析解が存在するテストケース（1チャンネルR-matrix、既知の光学模型）での絶対精度検証

### 7.2 性能受け入れ

- PERF-B1: AWS g5.xlarge (A10G) 上で30,000 Numerov積分 < 1秒
- PERF-B4: R-matrixフィット（16パラメータ、500データ点、500反復）< 30秒 CPU / < 5秒 GPU
- PERF-B6: 10,000核種バッチ < 2時間 (AWS p4d.24xlarge A100)
- CPUバックエンド単体でTALYS同等以上

### 7.3 OSS品質

- `cargo clippy --all-targets -- -D warnings` パス
- `cargo test --all-features` パス (CI常時実行)
- rustdocカバレッジ 100% (public API)
- mdbookユーザーガイド (理論・インストール・チュートリアル)
- CHANGELOG.md (Semantic Versioning)

---

## 付録A: 用語定義

| 用語 | 定義 |
|------|------|
| HF | Hauser-Feshbach統計模型。複合核反応の統計的記述法 |
| R-matrix | Wigner-Eisenbud配置空間分割に基づく反応理論 |
| OMP | Optical Model Potential。光学模型ポテンシャル |
| NLD | Nuclear Level Density。核レベル密度 |
| γSF | Gamma-ray Strength Function。γ線強度関数 |
| MACS | Maxwellian Averaged Cross Section。Maxwell平均断面積 |
| SEF | Stellar Enhancement Factor。恒星増強因子 |
| RIPL-3 | Reference Input Parameter Library v3 (IAEA) |
| REACLIB | JINA Reaction Rate Library |
| S因子 | クーロン障壁透過因子を除去した天体物理学的断面積因子 |

---

## 付録B: 検証参照実装

| モジュール | 参照実装 | 用途・注意事項 |
|-----------|---------|---------------|
| Coulomb関数 | 1. mpmath (50桁+)  2. COULCC.f (Fortran)  3. GSL | mpathが最高優先。GSLはη>>1かつρ<<ηで精度低下 |
| 透過係数 | TALYS v2.2 | 同一OMPパラメータでの比較 |
| HF断面積 | TALYS v2.2, CoH3 | 2コードでのクロスチェック推奨 |
| R-matrix | AZURE2, SAMMY 8.1 | 異なるR-matrix形式論の実装と比較 |
| 反応率 | NON-SMOKERデータベース, NACRE II | 公開データとの統計的比較 |

---

## 付録C: ベンチマーク問題定義

> **v0.2.0新規追加:** 性能目標の前提条件を明確化するため、代表的なベンチマーク問題を定義。

### B1: ⁵⁶Fe(n,γ)⁵⁷Fe 透過係数

球形中重核の代表的ケース。s過程の主要反応。

- 標的核: ⁵⁶Fe (Z=26, A=56, 球形)
- 入射粒子: 中性子
- エネルギーグリッド: 1 keV – 20 MeV, 1,000点（対数等間隔）
- 部分波: l = 0 – 30（収束判定: T_l < 10⁻¹⁰ で打切り）
- OMP: Koning-Delaroche (2003) 大域的パラメータ
- Numerovステップ: 0.05 fm（約400ステップ/積分）
- 総積分数: 1,000 × 31 = 31,000

### B2: ⁵⁶Fe(n,γ)⁵⁷Fe HF断面積

球形核のHF計算。粒子放出2段階 + γカスケード。

- B1と同一条件の透過係数を使用
- NLD: Gilbert-Cameron複合模型 (RIPL-3パラメータ)
- γSF: EGLO (Kopecky-Uhl)
- 粒子放出: 最大2段階 (n, p, α)
- γカスケード: 最大30ステップ
- J_max = 30, π = ±

### B3: ²³⁸U(n,γ)²³⁹U HF断面積

変形重核のストレステスト。結合チャンネル + 全γカスケード。

- 標的核: ²³⁸U (Z=92, A=238, 変形 β₂≈0.22)
- 結合チャンネル: 0+, 2+, 4+状態への結合
- 粒子放出: 最大5段階 + 全γカスケード
- 核分裂競合: 含む（Phase 2以降）

本問題はPERF-B2の1〜2桁遅い計算時間を許容する。主目的は変形核での正確性検証と、γカスケード状態空間爆発の実際の影響を計測すること。

### B4: ⁷Be(p,γ)⁸B R-matrixフィット

AZURE2のBRICK論文で使用された標準的なR-matrixフィッティング問題。

- 反応: ⁷Be(p,γ)⁸B（太陽ニュートリノ・原始核合成に重要）
- R-matrix準位数: 9
- 粒子対: 3 (⁷Be+p, ⁷Li+n, α+α)
- フィットパラメータ: 16（残りは固定）
- 実験データ点: 500
- LM法反復回数: 500回まで
- MCMC: 10,000サンプル × 32ウォーカー (別途B5で評価)

### B5: ¹²C(α,γ)¹⁶O R-matrix MCMC

核天体物理学の「聖杯」と呼ばれる反応のMCMCベンチマーク。

- 反応: ¹²C(α,γ)¹⁶O
- パラメータ: 16（E1/E2多極子、サブスレッショルド極含む）
- 実験データ: 1,000点（散乱+捕獲+β遅延αスペクトル）
- MCMC: Affine-invariant, 10,000サンプル, 32ウォーカー
- 目標: GPU 1基で30分以内 → 1サンプルあたり < 0.2秒

### B6: 全核図バッチ

TENDL反応率ライブラリ生成を想定した大規模バッチ計算。

- 核種数: 10,000（Z=10–83、中性子/陽子ドリップライン間）
- 入射粒子: n, p, α
- エネルギー: 1 keV – 20 MeV, 200点
- 全核種球形OMP + HF + MACS (60温度点)
- 出力: REACLIB形式 + HDF5テーブル

---

*— End of Document —*
