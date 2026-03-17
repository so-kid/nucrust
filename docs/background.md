# 宇宙元素合成における原子核反応率の計算機シミュレーション手法

恒星内元素合成の反応率計算は、**統計模型（Hauser-Feshbach）**、**R-matrix理論**、**直接反応模型（DWBA）**の3つの理論的枠組みを核として、光学模型ポテンシャル・レベル密度・γ線強度関数といった核物理入力と組み合わせることで実現される。これらの理論計算コード（TALYS, AZURE2, FRESCO等）で得られた反応率は、**REACLIB形式**の7パラメータフィットとしてデータベース化され、反応ネットワークソルバー（SkyNet, WinNet, XNet等）に取り込まれて元素合成シミュレーションに利用される。近年は**ベイズ推定**による不定性定量化や**機械学習**を用いた核物理量予測が急速に発展しており、FRIB等の新施設からの実験データと相まって、この分野は大きな転換期にある。

---

## 1. 原子核反応率計算の理論的枠組み

### 1.1 統計模型（Hauser-Feshbach模型）

Hauser-Feshbach（HF）模型は、1952年にW. HauserとH. Feshbachが提唱した複合核反応の統計的記述法であり（Phys. Rev. 87, 366）、中重核（**A ≳ 20–30**）の反応率計算における標準的手法である。核心となる仮定は「複合核の崩壊様式がその形成過程に依存しない」というBohrの独立性仮説であり、反応断面積は次式で与えられる：

$$\sigma(a,b) = \frac{\pi}{k_a^2} \sum_{J,\pi} \frac{(2J+1)}{(2j_a+1)(2J_A+1)} \frac{T_a(E,J,\pi) \cdot T_b(E,J,\pi)}{\sum_c T_c(E,J,\pi)}$$

ここで $T_c$ はチャネル $c$ の透過係数であり、この計算には3つの核物理入力が本質的に重要となる：**光学模型ポテンシャル**（粒子チャネルの $T_l$ を決定）、**核レベル密度**（連続準位への崩壊を記述）、**γ線強度関数**（光子チャネルの $T_\gamma$ を決定）。

HF模型の適用範囲は、複合核の励起エネルギーにおけるレベル密度が十分高い場合（おおむね各スピン・パリティあたり**5–10準位/MeV以上**）に限られる。軽い核（A < 20）や閉殻近傍核ではレベル密度が不足するため、R-matrix理論や直接反応模型が必要となる。Rauscher et al.（2002, ApJ 576, 323）は「ケイ素より重い元素では、レベル密度が十分高くHF模型が適用可能」と述べている。

**幅揺らぎ補正（Width Fluctuation Correction; WFC）** は、入射チャネルと出射チャネル間の相関を考慮する補正であり、特に弾性散乱の増強（elastic enhancement）を正しく取り扱うために重要である。Moldauer（1975, Phys. Rev. C 12, 744）やTepel-Hoffmann-Weidenmüller（1974）の形式論が用いられ、最新のTALYSやCoH3コードに実装されている。

### 1.2 R-matrix理論

R-matrix理論はWigner & Eisenbud（1947, Phys. Rev. 72, 29）が創始し、Lane & Thomas（1958, Rev. Mod. Phys. 30, 257）が体系化した配置空間分割に基づく反応理論である。空間をチャネル半径 $a$ で内部領域と外部領域に分け、内部領域の波動関数を正規直交基底で展開する：

$$R_l(E) = \sum_{n=1}^{N} \frac{\gamma_{nl}^2}{E_{nl} - E}$$

ここで $E_{nl}$ は極エネルギー、$\gamma_{nl}$ は換算幅振幅である。R行列から衝突行列 $U_l$ が導出され、断面積が計算される。1準位1チャネル近似ではBreit-Wigner公式に帰着する。

**チャネル半径 $a$** は、核力が無視できる境界として典型的に $a \approx (1.25\text{–}1.45)(A_1^{1/3} + A_2^{1/3})$ fm に設定される。個々のR-matrixパラメータは $a$ の選択に依存するが、物理量（断面積・位相差）は $a$ に依存しない。

実践上特に重要なのが**Brune変換**（2002, Phys. Rev. C 66, 044611）で、これは標準R-matrixと数学的に等価でありながら、準位シフトと境界条件定数を排除し、極位置が直接観測可能な共鳴エネルギーと一致する代替パラメータ化を提供する。これによりフィッティングが劇的に単純化される。

R-matrix理論は**軽い核系（A ≲ 20）** でのクーロン障壁以下へのS因子外挿、広い共鳴間の干渉効果の取り扱い、および ¹²C(α,γ)¹⁶O のような天体物理学的に重要な反応の解析に不可欠である。主要レビューとしてDescouvemont & Baye（2010, Rep. Prog. Phys. 73, 036301）がある。

### 1.3 直接反応模型（DWBA）

歪曲波ボルン近似（DWBA）は、入射粒子が少数の核子と短時間（～10⁻²² s）で相互作用する直接反応を記述する。遷移振幅は：

$$T_{fi} = \langle \chi_f^{(-)} \Phi_f | V | \Phi_i \chi_i^{(+)} \rangle$$

で与えられ、$\chi_{i,f}$ は光学ポテンシャルにより生成される歪曲波、$V$ は結合相互作用である。DWBAは**軽い核系**、**移行反応** (d,p), (³He,d) 等の解析、および**周辺反応**に適している。

移行反応の解析では**分光学的因子（spectroscopic factor）** $S_i$ を実験角度分布とDWBA計算の比較から抽出する。天体物理応用では、**漸近的規格化係数（ANC: Asymptotic Normalization Coefficient）** が周辺的な直接捕獲反応のS因子をモデル非依存に制約する手法として重要である（Texas A&M グループが先駆）。

### 1.4 光学模型ポテンシャル（OMP）

HF計算における粒子透過係数 $T_l(E)$ は光学模型ポテンシャルから算出される。現在の標準は**Koning-Delaroche（2003, Nucl. Phys. A 713, 231）** の大域的核子OMPであり、1 keVから200 MeVの範囲で24 ≤ A ≤ 209の球形核に適用可能なWoods-Saxon型ポテンシャルを提供する。実部体積項・虚部体積/表面項・スピン軌道項のすべてについて、エネルギー依存する深さと物理的に制約された幾何学パラメータが定められている。

α粒子については**McFadden-Satchler（1966）** が古典的標準であるが、**Avrigeanu et al.（2014, Phys. Rev. C 90, 044612）** の改良OMPや**ATOMKI-V2**ポテンシャル（SMARAGDで採用）がより精密な記述を提供する。荷電粒子反応（特にα誘起反応）ではクーロン障壁以下のエネルギーにおけるα-OMPの不定性が反応率の主要な誤差源となり、p過程元素合成において特に重要である。

### 1.5 核レベル密度モデル

核レベル密度（NLD）はHF計算の分母（全崩壊幅）と分子（出射チャネル幅）の双方に寄与し、特に中性子捕獲断面積に大きな影響を与える。安定核から離れた核種では、NLDの不定性が予測断面積を**2–10倍**変動させうる。

主要なモデルは以下の通りである。**逆移動フェルミガス模型（BSFG）** はBetheの公式にペアリング相関を補正する逆移動量 $\Delta$ を導入し、Dilg et al.（1973, Nucl. Phys. A 217, 269）が系統的パラメータを提供した。**定温度（CT）模型** は低励起エネルギーで準位の累積数が指数関数的に増加する経験則に基づく。**Gilbert-Cameron複合模型**（1965, Can. J. Phys. 43, 1446）は低エネルギーでCT模型、高エネルギーでBSFG模型を滑らかに接続する。**Ignatyuk**の公式（1975）は殻効果の励起エネルギーに伴う減衰を $a(U) = \tilde{a}(1 + \delta W / U \cdot [1 - e^{-\gamma U}])$ で記述し、TALYSの標準的アプローチとなっている。

最新の微視的アプローチとして、Goriely, Hilaire, Koningらによる**Skyrme HFB組み合わせ法** が大規模NLDテーブルを提供し、安定線から中性子ドリップ線までのすべての核種に対して一貫した予測を行う。TALYSでは6種類のNLDモデルが利用可能である。

### 1.6 γ線強度関数（γSF）

γ線強度関数は放射捕獲反応の断面積を決定する主要因子である。**標準ローレンツ型（SLO）** はBreit-Axel仮説に基づき巨大双極共鳴（GDR）のローレンツ分布で記述する。しかし球形核では低エネルギー強度を過大評価するため、**Enhanced Generalized Lorentzian（EGLO）**（Kopecky & Uhl, 1990）がエネルギー・温度依存幅を導入し、$E_\gamma \to 0$ での有限値を正しく再現する。

微視的**QRPA**計算（Goriely & Khan, 2002; Martini et al., 2016）はSkyrme/Gogny有効相互作用を用いて全核種のE1強度を計算し、特に中性子過剰核に現れる**ピグミー双極共鳴（PDR）** による低エネルギーE1強度の増強を予測する。これはr過程中性子捕獲率を大幅に増大させる可能性がある。

**upbend現象**（Oslo法で最初に観測、～2004年）は $E_\gamma \lesssim 2\text{–}4$ MeVにおけるγ線強度の増強であり、M1遷移が主因と考えられている。中性子分離エネルギーが低い不安定核（$S_n \lesssim 2$ MeV）ではupbend領域のみがγ崩壊に寄与するため、r過程捕獲率を**2–100倍**増強する可能性がある。

### 1.7 天体物理学的S因子

荷電粒子反応では、クーロン障壁によるトンネル効果の指数的な断面積減少を除去した**天体物理学的S因子**が定義される：

$$S(E) = \sigma(E) \cdot E \cdot \exp(2\pi\eta)$$

ここで $\eta = Z_1 Z_2 e^2 / (\hbar v)$ はSommerfeldパラメータである。S因子は非共鳴反応では緩やかにエネルギー変化するため、ガモフ窓エネルギーへの外挿に適している。ガモフピークエネルギーは $E_0 = (bk_BT/2)^{2/3}$ で与えられる。

外挿手法としては、**多項式外挿**（$S(E) = S(0) + S'(0)E + \frac{1}{2}S''(0)E^2 + \cdots$）が非共鳴反応に用いられるが、共鳴構造がある場合は**R-matrixベースの外挿**が最も信頼性が高い。AZURE2コードがこの目的の現代的標準ツールである。

---

## 2. 主要な計算コード・ソフトウェア

### 2.1 TALYS — 統計模型コードの決定版

**TALYS** は200 MeV以下の核反応シミュレーションのための包括的コードであり、質量数12 ≤ A ≤ 339の核種を対象とする。Hauser-Feshbach複合核模型、ECIS-06統合による結合チャネル直接反応、2成分エキシトン前平衡模型、多モード核分裂を実装している。

| 項目 | 詳細 |
|------|------|
| **言語** | Fortran（gfortran対応） |
| **ライセンス** | MIT（オープンソース） |
| **最新版** | TALYS-2.2（GitHub上でベータ版も継続更新） |
| **GitHub** | https://github.com/arjankoning1/talys |
| **IAEA** | https://www-nds.iaea.org/talys/ |
| **マニュアル** | https://nds.iaea.org/talys/tutorials/talys.pdf |
| **ディスク容量** | 約8 GB（構造データベース含む） |

入力ファイルはキーワードベースの平文形式で極めて簡潔である。最小構成例：

```
projectile n
element Fe
mass 56
energy energies.txt
```

**天体物理モードの起動**には `astro y` キーワードを設定する。これにより以下が自動計算される：Maxwell平均断面積（MACS）、天体物理反応率 $N_A\langle\sigma v\rangle$、荷電粒子反応のS因子、熱的励起標的を考慮した恒星増強因子。出力ファイルには `astrorate.g`, `astrorate.tot` 等が生成される。関連論文はGoriely, Hilaire, Koning（2008, A&A 487, 767）。

TALYSに基づく系統的反応率データとして**TENDL-astro**（https://tendl.web.psi.ch/tendl_2023/astro/astro.html）が約8,892核種の反応率を提供している。関連ツールとして不定性解析用の**TASMAN**（https://github.com/arjankoning1/tasman）、実験データ比較用の**EXFORTABLES**（https://github.com/arjankoning1/exfortables）がある。

### 2.2 NON-SMOKER / SMARAGD

Rauscher & Thielemann（2000, ADNDT 75, 1）による天体物理反応率に特化したHF計算コードである。TALYSとの主な違いは、**低エネルギー天体物理領域に最適化**されている点、Lejeune微視的中性子光学ポテンシャルの採用、前平衡過程の非包含などである。NON-SMOKERの後継として**SMARAGD**（v0.42.0s, 2025年12月時点）が開発されているが、スタンドアロンコードとしての公開配布はされておらず、著者Thomas Rauscherへの連絡が必要である。

NON-SMOKERデータベースは https://nucastro.org/nonsmoker.html で無料公開されており、Webインターフェース（NON-SMOKERweb）は無料登録で利用可能（https://nucastro.org/websmoker.html）。

### 2.3 AZURE2 — R-matrixフィッティングコード

AZURE2は低エネルギー核反応のR-matrix解析・外挿のための多チャネル・多準位コードであり、Notre Dame大学/JINA-CEEで開発されている。Lane-Thomas形式のR-matrix理論を実装し、散乱・捕獲・反応断面積の同時フィットが可能。MINUIT2（CERN）による最小二乗フィッティングエンジンを搭載し、GUIモードとコマンドラインモード（`--no-gui`）の両方で動作する。

| 項目 | 詳細 |
|------|------|
| **言語** | C++（元のAZUREはFortran、AZURE2は完全書き直し） |
| **依存** | CMake ≥ 2.8, GSL, Minuit2, Qt（GUI用、任意）, Qwt（描画用、任意） |
| **GitHub** | https://github.com/rdeboer1/AZURE2 |
| **参考論文** | Azuma et al., Phys. Rev. C 81, 045805 (2010) |

ベイズ推定のための連携ツールとして**BRICK**（Bayesian R-matrix Inference Code Kit）が開発されており、Pythonのemcee MCMCサンプラーとAZURE2を組み合わせてR-matrixパラメータの事後分布を推定する（Odell et al., Front. Phys. 10, 888476, 2022）。

### 2.4 EMPIRE, FRESCO, CoH3, SAMMY

**EMPIRE 3.2.3**（BNL/NNDC、Fortran/C）はTALYSと同等の機能に加え、**重イオン入射**が可能な点が大きな特長である。IAEA（https://www-nds.iaea.org/empire/）やRSICC経由で入手可能。

**FRESCO/FRESCOX**（Ian Thompson, Fortran 90/95）は結合チャネル・DWBA・CRC・CDCCのための包括的コードである。名前リストベースの入力形式を持ち、移行反応断面積の計算を通じて分光学的因子やANCの抽出に用いられる。GitHub上でオープンソース公開されている（https://github.com/I-Thompson/fresco, https://github.com/LLNL/Frescox）。

**CoH3**（河野俊彦, LANL, C++）は光学模型+HFコードで、2024年10月にGitHubで公開された（https://github.com/toshihikokawano/coh3）。

**SAMMY 8.1.0**（ORNL, Fortran）はReich-Moore近似によるR-matrix解析コードであり、ベイズ推定によるパラメータフィッティング、ドップラー広がり補正、分解能関数の畳み込みを含む。**最近のENDFの分解共鳴データの約80%はSAMMYで生成**されている。ORNL（https://www.ornl.gov/onramp/sammy）やNEA Data Bankで配布。

---

## 3. 反応率データベースとREACLIB形式

### 3.1 JINA REACLIBデータベース

**JINA REACLIB**（https://reaclib.jinaweb.org/）は天体物理反応率の最も広く使われるデータベースであり、Cyburt et al.（2010, ApJS 189, 240）に記述されている。全反応率は**7パラメータフィット公式**で表現される：

$$\lambda = \exp\left[a_0 + \frac{a_1}{T_9} + \frac{a_2}{T_9^{1/3}} + a_3 T_9^{1/3} + a_4 T_9 + a_5 T_9^{5/3} + a_6 \ln T_9\right]$$

ここで $T_9$ はGK単位の温度である。複数の7パラメータセットの和として共鳴・非共鳴成分を表現でき、フィット温度範囲で**5%以内**の精度を目標とする。

**REACLIB形式の詳細仕様**（R1形式）は以下の通りである。各行は正確に**74文字**で、1つの反応率エントリは**3行**から成る：

- **1行目**: 5スペース + 核種名6フィールド（各5文字） + 8スペース + ラベル4文字 + フラグ1文字（'n'=非共鳴, 'r'=共鳴, 'w'=弱い相互作用） + 逆反応フラグ1文字（'v'=詳細つり合いによる逆反応） + 3スペース + Q値12文字
- **2行目**: $a_0$–$a_3$ の4フィールド（各13文字）
- **3行目**: $a_4$–$a_6$ の3フィールド（各13文字）

**Chapter番号**（1–11）は反応タイプを示す：Chapter 1 = 崩壊 $e_1 \to e_2$、Chapter 4 = 捕獲 $e_1+e_2 \to e_3$、Chapter 5 = $e_1+e_2 \to e_3+e_4$、Chapter 8 = 3体反応（トリプルアルファ型）$e_1+e_2+e_3 \to e_4$。

Fortran読み込み書式は：
```fortran
format(i1,4x,6a5,8x,a4,a1,a1,3x,1pe12.5)
format(4e13.6)
format(3e13.6)
```

Pythonでの読み込みには**pynucastro**（https://github.com/pynucastro/pynucastro）が最も便利である：
```python
import pynucastro as pyna
rl = pyna.ReacLibLibrary()
c13pg = rl.get_rate_by_name("c13(p,g)n14")
c13pg.eval(1.e9)  # 1 GKで評価
```

### 3.2 STARLIB — Monte Carloベース不定性付き反応率

STARLIB（Sallaska et al., 2013, ApJS 207, 18; https://starlib.github.io/Rate-Library/）は、Longland/Iliadisの**Monte Carlo法**によって評価された約80の実験反応率を含み、各温度グリッド点で反応率の確率密度関数（PDF）を提供する唯一のライブラリである。反応率分布は**対数正規分布**で近似され、推奨値（中央値）と因子不定性（factor uncertainty）で特徴づけられる。REACLIB形式ではなく**表形式**（温度 vs 反応率）であり、不定性情報（0.16および0.84分位点）が含まれる。

### 3.3 KADoNiS、BRUSLIB/NETGEN

**KADoNiS**（https://www.kadonis.org/）はKarlsruhe研究所のs過程向け中性子捕獲断面積データベースであり、kT = 5–100 keVにおけるMACSを357核種以上について提供する。**BRUSLIB**（http://www.astro.ulb.ac.be/bruslib/）はブリュッセル自由大学のライブラリで、NACRE/NACRE IIの実験ベース反応率とTALYSベースの約100,000のHF理論反応率をBrussels-Montreal Skyrme-HFB模型で一貫して計算した点が特長である。対応するネットワーク生成ツール**NETGEN v10.0**はWeb上で対話的にネットワーク構築が可能。

---

## 4. 反応ネットワーク計算の数値手法

### 4.1 硬い（stiff）ODE系の数値解法

核反応ネットワークの方程式系 $dY_i/dt = f_i(\{Y_j\})$ は、関与する反応のタイムスケールが**10¹⁵倍以上**異なる典型的なstiff系である（爆発的燃焼時のナノ秒スケールの反応から、10⁶年スケールの放射性崩壊まで）。このため陽的解法（前方Euler, RK4等）は実用的でなく、**陰的解法**が必須となる。

**後退Euler法**は最も基本的な陰的解法で、$Y^{(n+1)} = Y^{(n)} + \Delta t \cdot f(Y^{(n+1)})$ を各ステップでNewton-Raphson反復により解く。**BDF法（後退微分公式）**は複数の過去ステップの情報を用い、5次までの可変次数法として実装される。**Gear法**はBDFの可変次数・可変ステップ幅の自動適応実装であり、WinNetではimplicit EulerとGear法の両方が実装されている。Gear法は滑らかな進化では効率的だが、条件の不連続変化に対してはimplicit Eulerがよりロバストであると報告されている。

**ヤコビ行列の疎性** はネットワーク計算の効率化において決定的である。487核種・5,892反応のネットワークではヤコビ行列の**96.4%が零要素**であり（西村、RIKEN発表）、疎行列ソルバー（PARDISO, UMFPACK, Intel MKL等）を用いることで計算コストを $O(N^3)$ から実質 $O(N)$–$O(N^2)$ に低減できる。

**核統計平衡（NSE）** は温度 $T \gtrsim 5\text{–}7$ GK で強い相互作用の前方・逆方向反応が平衡に達する状態であり、存在比はSaha方程式で代数的に決定される。NSEからの離脱（freeze-out）の数値的取り扱いが技術的課題であり、SkyNetではNSEと非平衡ネットワーク進化の自己矛盾なき切り替えアルゴリズムが実装されている。

**適応的ネットワークサイズ**は、存在比や反応フローが閾値を超える核種のみを動的に含めることで計算効率を向上させる。NuGridのPPNコードはセルレベルで局所熱力学環境に応じたネットワークサイズを自動生成する。

### 4.2 主要ネットワークソルバーの比較

| コード | 言語 | 解法 | リポジトリ | 特徴 |
|--------|------|------|-----------|------|
| **XNet** | Fortran 90/95 | Backward Euler + Newton-Raphson | https://github.com/starkiller-astro/XNet | FLASH/CHIMERAとの結合利用、NSE/QSE対応、OpenMP |
| **SkyNet** | C++ + Python | Implicit solver + PARDISO | https://bitbucket.org/jlippuner/skynet | r過程特化、自己加熱、Helmholtz EOS、NSE自動切替 |
| **WinNet** | Fortran | Implicit Euler + Gear法 | https://github.com/nuc-astro/WinNet | 5800+核種、多様な弱い反応率・核分裂データ対応 |
| **libnucnet** | C + Python | — | https://sourceforge.net/p/libnucnet/ | XML形式データ管理、XPathクエリ、教育用途 |
| **pynucastro** | Python → C++/Python | — | https://github.com/pynucastro/pynucastro | ネットワーク生成・可視化、AMReX連携、Jupyter対応 |

SkyNetは**7,800核種・140,000反応**規模のネットワークに対応し、熱力学軌道に沿った元素合成計算や自己加熱計算が可能である。入力はREACLIB形式の反応率ファイル、核種リスト、温度・密度・電子割合の時間発展であり、出力はHDF5またはテキスト形式の時間依存存在比ベクトルである。WinNetの文書化は https://nuc-astro.github.io/WinNet/ で公開されており、多数の例題計算（原始核合成、恒星燃焼、超新星、中性子星合体）が付属している。

### 4.3 Monte Carloによる反応率不定性の伝搬

Longland et al.（2010, Nucl. Phys. A 841, 1）およびIliadis et al.（2010a,b,c）が確立したMonte Carlo法は、核物理入力量（共鳴強度、エネルギー、部分幅）にそれぞれ適切な確率分布（ガウス分布、対数正規分布、Porter-Thomas分布等）を割り当て、**10,000回以上のサンプリング**により各温度での反応率分布を統計的に特徴づける。結果として得られる反応率PDFは**対数正規分布**で良く近似され、中央値（推奨率）と $\sigma$（因子不定性）で記述される。

元素合成計算への伝搬は、各反応率に対数正規分布からサンプリングした乗算因子 $p_i$ を適用し、多数の元素合成計算を実行することで行う。最終存在比と $p_i$ の相関解析により、どの反応が出力に最も影響するかを特定できる。Longland（2017, A&A 604, A34）は相関不定性への拡張も提案している。

---

## 5. 天体物理シミュレーションとの連携

### 5.1 MESAの核反応ネットワーク

**MESA**（https://mesastar.org/; Paxton et al. 2011–2023の6本のinstrument paper）は1次元恒星進化コードの事実上の標準であり、Fortran 95で記述されOpenMPで並列化されている。核反応ネットワークは2つのモジュールで提供される：

- **`net`モジュール**: 事前定義されたネットワーク定義を使用。`basic.net`（最小水素燃焼）から`approx21.net`（21核種α鎖）まで多数のオプションがある。
- **`jina`モジュール**: JINA REACLIBデータベースを直接使用し、ユーザー指定の核種リストから自動的にすべての関連反応を含む大規模ネットワークを構築する。76,000以上の反応・4,500以上の核種をカバー。

カスタム反応率の追加は `star_job` ナメリストで `change_net = .true.` と設定し、`$MESA_DIR/data/rates_data/` に反応率ファイルを配置することで行える。反応率はNACRE、JINA REACLIB、STARLIBから選択可能で、個別反応ごとにどの評価を使用するか指定できる。

### 5.2 NuGridとNuPyCEE

**NuGrid**（https://nugrid.github.io/）のPPNコードは多ゾーン・混合付きの核合成計算コードであり、5,000以上の核種・50,000以上の反応を扱い、MPIで並列化されている。HDF5ベースのUSEEPPライブラリでデータI/Oを管理する。**NuPyCEE**（https://github.com/NuGrid/NuPyCEE）はPython 3による銀河化学進化ツール群であり、SYGMA（単純星群コード）、OMEGA（銀河化学進化モデル）、STELLAB（観測恒星元素存在比ライブラリ）を含む。

### 5.3 典型的ワークフロー

実装上の典型的なワークフローは以下の通りである：

1. **実験** → 加速器施設（LUNA, JUNA, CASPAR, Felsenkeller等の地下実験室、またはRIKEN RIBF, FRIBのRIビーム施設）で断面積を測定
2. **R-matrixフィット** → AZURE2コードで実験データにR-matrixパラメータをフィット
3. **不定性定量化** → BRICK（AZURE2 + emcee MCMC）でベイズ事後分布を推定
4. **反応率計算** → RatesMC（Monte Carlo法）で核物理不定性を反応率PDFに伝搬、またはTALYSでHF反応率を計算
5. **REACLIB形式への変換** → 7パラメータフィットを対数空間での最小二乗法で決定
6. **ネットワーク計算** → SkyNet/WinNet/MESA等にREACLIB/STARLIB形式で取り込み、元素合成シミュレーションを実行
7. **観測との比較** → 太陽系存在比、恒星表面元素存在比、キロノヴァ光度曲線等と比較

---

## 6. 近年の発展：機械学習とベイズ推定

### 機械学習による核物理量予測

近年、核質量予測においてML手法が急速に発展している。**Mumpower et al.（2022, Phys. Rev. C 106, L021301）** は物理動機付き特徴量を用いた確率的ニューラルネットワークでrms偏差186 keV（学習データ）/316 keV（全AME, Z≥20）を達成した。**Neufcourt et al.（2018, Phys. Rev. C 98, 034318）** のベイズガウス過程法は外挿性能に優れ、r過程シミュレーションにML質量を初めて投入した研究（Neufcourt et al., 2023, Phys. Lett. B）では存在比ピークの位置が良好に再現された。2025年には多タスクガウス過程による質量（**rms 0.136 MeV**）と電荷半径（rms 0.007 fm）の同時予測も報告されている。反応断面積への直接適用としては、XGBoostによる(n,2n)断面積予測（2025, Nucl. Sci. Eng.）がENDF/Bの物理パラメータを用いて415核種で訓練され、エキゾチック核への外挿を実現している。

### ベイズ推定によるR-matrixパラメータ決定

**BRICK**（Odell et al., 2022, Front. Phys. 10, 888476）はAZURE2とPythonのemcee MCMCサンプラーを連携させ、R-matrixパラメータのベイズ推定を可能にした。系統的不定性の任意の相関構造を取り扱い、ab initio計算からの制約（ANC等）を事前分布として組み込める。³He(α,γ)⁷Be の解析に適用され、従来の頻度論的手法では困難であった不定性の完全な伝搬を実現した。

### FRIBの影響

**FRIB**（Facility for Rare Isotope Beams）は2022年5月に運用を開始し、2024年時点で270以上の希少同位体ビームを提供している。2023年にはTm-182, Tm-183, Yb-186, Yb-187, Lu-190といった中性子星衝突でのみ存在すると理論予測されていた核種の生成に成功した。2023年12月には世界記録の10.4 kWウラン-238ビームでGa-88, As-93, Se-96を発見。r過程第3ピーク（A≈195）に関わるN=126閉殻近傍の核構造研究が進行中であり、質量・半減期・遅発中性子放出確率の測定がr過程モデルの制約を大幅に改善しつつある。

---

## 7. 日本の貢献と日本語リソース

**理研RIBF**（仁科加速器科学研究センター）はFRIB以前の世界最先端RI（放射性同位体）ビーム施設であり、超伝導リングサイクロトロン（SRC）で²³⁸Uを光速の70%まで加速する。**EURICA共同実験**（Lorusso, Nishimura, Sakurai et al., PRL 2015）ではN=82閉殻近傍の中性子過剰核110核種の半減期を測定し、うち40核種は世界初測定であった。これはr過程第2ピークと希土類元素の元素合成理解に直結する成果である。

**JCPRG**（日本荷電粒子核反応データグループ、https://www.jcprg.org/）はEXFORを補完する**NRDF/A**（天体物理向け核反応データファイル）を維持管理している。**JAEA**のJENDL-5（岩本 et al., 2023）は天体物理向けMACS計算にも利用されている。**大阪大学RCNP**の川端グループはトリプルアルファ反応率の実験的決定に貢献している。

RIKEN RIBFでは定期的にミニワークショップが開催されており、2023年2月の「星の進化と爆発天体における核反応の物理」（MiniWS039）では不安定核の構造・反応機構から天体核反応まで幅広い議論がなされた。

日本語の教科書としては、朝倉書店の「原子核物理学」（基礎物理学シリーズ）が元素合成の章（元素の誕生）でガモフ因子、s過程、r過程を解説している。ただし、核天体物理学に特化した日本語の専門教科書は確認されておらず、多くの日本人研究者はIliadis, Clayton, Rolfs & Rodneyの英語教科書を使用している。

---

## 8. 主要文献と参考リソース

### レビュー論文・教科書

| 文献 | 内容 |
|------|------|
| Iliadis, *Nuclear Physics of Stars*, 2nd ed. (Cambridge, 2015) | 核天体物理学の標準教科書 |
| Clayton, *Principles of Stellar Evolution and Nucleosynthesis* (1968) | 恒星物理・元素合成の古典 |
| Rolfs & Rodney, *Cauldrons in the Cosmos* (1988) | 核天体物理学の先駆的教科書 |
| Thompson & Nunes, *Nuclear Reactions for Astrophysics* (Cambridge, 2009) | 反応理論のR-matrix/DWBA/結合チャネルと天体物理応用 |
| Descouvemont & Baye, Rep. Prog. Phys. **73**, 036301 (2010) | R-matrix理論の包括的レビュー |
| Rauscher & Thielemann, ADNDT **75**, 1 (2000) | HF反応率テーブル（Z=10–83） |
| Cowan et al., Rev. Mod. Phys. **93**, 015002 (2021) | r過程の最新総合レビュー |
| Käppeler et al., Rev. Mod. Phys. **83**, 157 (2011) | s過程の総合レビュー |
| deBoer et al., Rev. Mod. Phys. **89**, 035007 (2017) | ¹²C(α,γ)¹⁶Oの決定版レビュー |
| Angulo et al., Nucl. Phys. A **656**, 3 (1999) | NACREコンパイレーション |
| Cyburt et al., ApJS **189**, 240 (2010) | JINA REACLIBデータベース記述 |

### データベース・ポータルURL一覧

| リソース | URL |
|----------|-----|
| nucastrodata.org（総合ポータル） | https://nucastrodata.org/datasets.html |
| JINA REACLIB | https://reaclib.jinaweb.org/ |
| STARLIB | https://starlib.github.io/Rate-Library/ |
| KADoNiS | https://www.kadonis.org/ |
| BRUSLIB/NETGEN | http://www.astro.ulb.ac.be/bruslib/ |
| EXFOR | https://www-nds.iaea.org/exfor/ |
| NNDC | https://www.nndc.bnl.gov/ |
| TENDL | https://tendl.web.psi.ch/ |
| IAEA NDS | https://www-nds.iaea.org/ |
| AME2020 | https://www-nds.iaea.org/amdc/ |

---

## 結論：実装に向けた実践的指針

本報告で調査した枠組みを実装に活用する際の指針を以下に整理する。**核種の質量数で理論手法を選択する**のが第一歩であり、A ≲ 20ではR-matrix（AZURE2）やDWBA（FRESCO）、20 ≲ A ≲ 40では両手法の併用、A ≳ 40ではHF統計模型（TALYS）が適切である。反応率計算の結果は**REACLIB 7パラメータ形式**に統一的にフィットし、ネットワークソルバーに投入する。r過程計算にはSkyNetまたはWinNetが即座に利用可能で、恒星進化と連成した計算にはMESAが最も成熟したエコシステムを提供する。

不定性評価は現代の元素合成研究において不可欠であり、STARLIBのMonte Carlo反応率PDFを用いた不定性伝搬、またはBRICKによるベイズR-matrix解析が推奨される。JINA REACLIBの理論反応率の大部分が2009年以前のNON-SMOKER計算に基づいており更新が滞っている点（Smith, 2023, Front. Astron. Space Sci.）は、新しいTALYS 2.2やSMARAGDの計算結果でライブラリを更新する必要性を示唆している。機械学習とベイズ統計の融合は、FRIB等の新施設から得られる膨大な実験データと理論モデルを橋渡しする強力なツールとなりつつあり、今後数年でこの分野のパラダイムが大きく変わる可能性が高い。