
> **遥感 OVSS 的瓶颈之一并不一定在视觉编码器，而在“类别如何用语言表示”。**

论文题目是 **Reducing Semantic Ambiguity in Open-Vocabulary Remote Sensing Image Segmentation via Knowledge Graph-Enhanced Class Representations**，发表于 *ISPRS Journal of Photogrammetry and Remote Sensing*。其核心方法 KG-OVRSeg 用知识图谱重构类别语义表示，再利用类别感知解码器把这种语义信息注入像素预测。作者报告 7 个数据集平均 **51.65% mF1 / 39.18% mIoU**，相对第二名提高 8.06 / 6.52 个百分点。

但从你当前研究的角度，我认为更重要的是：

**这篇论文实际上为“Prompt Sets Matter”提供了一个非常强的、已经发表在 ISPRS JPRS 上的旁证；同时它也给你的 DAG Prompt Search 暴露了一个目前还可以继续增强的方向——“层级语义关系 + 跨类别安全约束”。**

---

# 一、这篇论文究竟发现了什么问题？

作者没有把问题定义为传统意义上的 domain gap，而是定义成了 **semantic ambiguity，语义歧义**。

这个定位其实很聪明。

遥感类别不像自然图像里的 `cat / dog / bicycle` 那么清晰，遥感标签天然带有 taxonomy / ontology 属性。

论文把问题分成三种。

### 1. One-to-Many：一个标签对应很多视觉概念

例如：

`impervious surface`

实际上可能包含：

`road / parking lot / roof / court ...`

所以你把：

> impervious surface

送进 CLIP，并不能保证这个单一词向量覆盖所有真实视觉形态。

论文后面的 Fig.14 非常有说服力：`impervious surface` 本身的激活可能并不好，但 `road`、`court`、`artificial facility` 等子概念却能明显激活正确区域。

这和你之前在 SAM3 上观察到的现象几乎是同一个本质：

> **dataset class label ≠ best model query。**

---

### 2. Many-to-One：不同词其实指同一种地物

比如论文指出：

`rangeland ↔ pasture`

`barren ↔ bareland`

`cropland ↔ plowed land`

虽然人类知道它们高度相关，但是 CLIP embedding 不一定靠得很近。

这说明：

**“label name”只是数据集作者采用的一种命名方式，并不是视觉语言模型的最优语义接口。**

作者明确指出，遥感 OVSS 中存在命名不一致、层级关系和 embedding semantic proximity 不足的问题。

---

### 3. Weak Semantic Proximity

更深一层的问题是：

两个本来高度接近的概念，在 CLIP text space 里面可能并不接近。

因此问题不仅是：

> “有没有更多 prompt？”

而是：

> **“这些 prompt 在模型自己的语义空间里到底形成了什么结构？”**

这一点其实和你的 RPS-SAM3 更接近。

---

# 二、KG-OVRSeg 的方法到底怎么做？

整个方法可以概括成：

$$
\text{Class Name}
\rightarrow
\text{Knowledge Subgraph}
\rightarrow
\text{Enhanced Class Embedding}
\rightarrow
\text{Class-aware Visual Decoding}
\rightarrow
\text{Segmentation}.
$$

论文图 3 很清楚：两个预训练 encoder 冻结，类别走 KGCE，图像走 image encoder，之后进行 feature fusion，再通过 CAGD 多阶段解码。

需要特别强调：

**它并不是 training-free 方法。**

虽然 CLIP 图像/文本 encoder 是 frozen 的，但后面的 decoder、CAA 等包含 learnable parameters，而且作者明确使用 OpenEarthMap 训练集训练 80k iterations。

这一点对于后面判断“你和它是否冲突”非常重要。

---

# 三、最重要的模块：KGCE

KGCE = **Knowledge Graph-enhanced Class Encoder**。

我认为这是整篇论文最值得你研究的部分。

## 3.1 先不给类别直接编码，而是查一个局部知识子图

对于一个类别：

$$
c_i
$$

不直接：

$$
E_i=\text{CLIP}(c_i)
$$

而是先查询：

$$
G_i=(V_i,R,T_i)
$$

也就是围绕这个类别找出一群相关节点。

论文采用的 LCCKG 有：

**383 个 land-cover classes、4 类关系、146,306 个 triples**，知识来自 127 个公开遥感语义分割数据集以及文献、已有 KG，并且部分层级关系由领域专家标注。

所以：

`impervious surface`

不再只是一个 token，而可能对应类似：

```text
impervious surface
 ├─ road
 ├─ road surface
 ├─ paved area
 ├─ court
 ├─ artificial facility
 ├─ roof
 └─ ...
```

这和你现在的 DAG Prompt Search 在形式上已经非常接近。

---

# 四、论文最值得借鉴的点之一：Global Filtering

这里我认为甚至比“Knowledge Graph”本身还重要。

作者意识到：

**知识越多不一定越好。**

例如数据集同时包含：

```text
agricultural land
vineyard
plowed land
```

如果 agricultural land 的子图里又包含：

```text
vineyard
plowed land
```

那么三个 class representation 就会严重重叠。

于是作者做了一个 **global filtering**。

如果：

$$
c_i \supset c_j
$$

同时：

$$
V_j\subseteq V_i,\quad
T_j\subseteq T_i
$$

就从父类别 \(G_i\) 中去掉子类别 \(G_j\) 对应的节点和关系。

这件事情特别重要。

因为它的思想已经不是：

> “给每一个 class 独立扩展 prompt。”

而是：

> **“Prompt expansion 必须考虑当前整个 dataset vocabulary。”**

这恰恰是我认为你现有 DAG Prompt Search 最值得进一步吸收的地方。

---

# 五、这对你当前 DAG Prompt Search 的启示非常大

你现在如果是：

```text
road
 ├─ paved road
 ├─ traffic lane
 ├─ arterial road
 └─ road surface

bareland
 ├─ bare soil
 ├─ exposed earth
 └─ barren land
```

这还主要是 **intra-class candidate expansion**。

KG-OVRSeg 给你的启发是应该再增加一个：

## Dataset-level cross-class graph reasoning

例如某个数据集同时存在：

```text
low vegetation
tree
forest
agriculture
```

那么一个 prompt：

```text
green vegetation
```

虽然对 `low vegetation` 响应可能很好，但它同时也可能：

```text
→ tree
→ forest
→ agriculture
```

都产生强响应。

因此它不能只按照：

> “对目标类响应强不强”

判断，而要考虑：

> **“它是否侵入其它类的语义子空间？”**

这实际上可以成为你现有 **Cross-Class Safety** 的理论依据。

---

# 六、KGCE 后面怎么形成最终 embedding？

这个部分反而很简单。

作者对知识子图中的每个节点分别使用 CLIP text encoder：

$$
h_j=\text{CLIPTextEncoder}(v_j)
$$

然后直接：

$$
E_i^T=\operatorname{AvgPool}(H_i).
$$

也就是：

## 所有知识节点直接平均。

论文方法明确如此。

这一点我要特别提醒：

### 这是论文的优点，同时也是你最不应该照搬的地方。

为什么？

因为论文自己在 Fig.14 中已经证明：

> **不同 subclass / synonym 对图像的响应差异非常大。**

例如 bare soil 里面，有些词基本没什么作用，但：

`sand barrier`

`sand`

响应反而更强。`grassland`、`shrub wood` 有时也比数据集原始标签表现更好。

既然不同节点的 visual response 差异这么大：

$$
\frac{1}{N}\sum_j e_j
$$

严格来说就不是最优解。

这恰恰给你的方法留下了空间。

---

# 七、你可以比它更进一步：Knowledge Proposal + Response Selection

这是我认为对你当前工作最有价值的一点。

KG-OVRSeg：

```text
KG
 ↓
所有相关词
 ↓
average
 ↓
class embedding
```

而你的路线完全可以变成：

```text
Class anchor
       ↓
Semantic / Knowledge DAG
       ↓
candidate prompts
       ↓
Frozen SAM3 response evidence
       ↓
cross-class safety
       ↓
reliability / discriminability pruning
       ↓
Frozen PromptBank
```

也就是：

> **KG-OVRSeg 是 knowledge-centric。**
>
> **你的路线应该继续坚持 model-response-centric。**

知识图谱负责：

**提出候选。**

SAM3 本身负责：

**决定候选是否可信。**

这个逻辑比直接平均所有 KG 节点，我认为更加适合 frozen SAM3。

---

# 八、论文一个非常有价值的实验：synonym 其实并没有想象中重要

Table 5 是整篇论文里我认为对你的 PromptBank 最有价值的表之一。

作者分别测试：

| Knowledge relation |  mean mF1 | mean mIoU |
| ------------------ | --------: | --------: |
| synonym            |     45.24 |     33.87 |
| contain            |     49.48 |     37.30 |
| synonym + contain  | **51.65** | **39.18** |



这个结果很关键。

说明：

## 单纯 synonym expansion 不够。

事实上，**hierarchical relation 的价值比 synonym 大得多。**

作者也明确指出，仅用 synonym 最差；contain 关系使 mean mF1/mIoU 又提升约 4.24/3.43 个百分点，二者联合最好。

这对你的 DAG 是一个很直接的启发。

你现在的 candidate operation 不应该只强调：

```text
Synonym Replacement
Attribute Addition
Scene-Context Refinement
```

我建议进一步明确增加：

```text
Hyponym Expansion
Visual Subclass Expansion
```

甚至比 generic attribute addition 更重要。

比如：

```text
impervious surface
→ road
→ court
→ paved surface
→ roof
```

不是普通的 synonym，而是 **visual decomposition**。

---

# 九、甚至可以重新理解你 DAG 的边

以后你的 DAG edge 不必只是：

```text
synonym
attribute
context
```

而可以变成一个真正具有语义类型的 DAG：

| Relation      | 含义   | 对 Prompt Search 的作用         |
| ------------- | ---- | --------------------------- |
| synonym       | 同义词  | 修复 naming mismatch          |
| hyponym       | 子概念  | 覆盖 one-to-many visual modes |
| hypernym      | 上位概念 | 增加泛化，但高风险                   |
| attribute     | 外观属性 | 增强视觉可识别性                    |
| scene context | 场景语义 | 增强遥感域解释                     |
| exclusion     | 排他关系 | 防止跨类 contamination          |
| redundancy    | 重复关系 | Prompt pruning              |

于是你的 PromptBank 就从：

> candidate phrase set

进一步变成：

> **structured semantic search space。**

这会明显增强方法层面的解释性。

---

# 十、CAGD 也有一个非常值得注意的设计

KGCE 得到 enriched class embedding 后，作者没有仅仅计算一次 cosine similarity。

而是做 **Class-aware Attention + Gradual Decoder**。

其中最有意思的是 Q/K/V 的方向。

作者实验了：

| Attention                 |  mean mF1 | mean mIoU |
| ------------------------- | --------: | --------: |
| Cross-attention           |     42.05 |     31.27 |
| text = Q, image = K/V     |     45.30 |     34.33 |
| **image = Q, text = K/V** | **48.52** | **36.77** |



作者的解释也很合理：

当：

$$
Q=Image,\quad K,V=Text
$$

等价于每一个 spatial position 在问：

> **“我更像哪个 class？”**

而不是：

$$
Q=Text
$$

让一个 class 去全局寻找图像。

因此 spatial localization 更好。

---

# 十一、这个思想对 SAM3 有没有直接借鉴意义？

**机制层面有，结构层面不能直接照搬。**

SAM3 本身已经有自己的 text-conditioned segmentation / query mechanism。

如果为了复现 CAA 再额外加 learnable：

```text
WQ
WK
WV
decoder
```

就会破坏你：

> Frozen SAM3 / training-free

这个核心卖点。

因此真正值得借鉴的是：

### “Class-conditioned competition”

而不是它的具体 CAA。

例如你完全可以在 frozen response level 做：

$$
L_c(x)
=
\operatorname{Aggregate}_{p\in P_c}
R(x,p)
$$

然后通过不同 class 的 logits：

$$
\{L_1(x),...,L_C(x)\}
$$

进行跨类别竞争。

也就是把 KG-OVRSeg 的：

> class-aware attention

转化成你的：

> **class-aware response competition / class-logit control。**

这样更加符合 SAM3 原生框架。

---

# 十二、GR 模块为什么有效？

GR = Guidance Refine。

本质上就是：

```text
high-level semantic map
        ↓
upsampling
        +
low-level class-conditioned spatial feature
        ↓
conv refinement
```

多阶段逐渐恢复空间细节。

它的增益很明显。

Table 3 中：

baseline：

$$
27.88\% \text{ mean mIoU}
$$

加入 GR：

$$
32.59\%
$$

增加约：

$$
+4.71
$$

再加入 CAA：

$$
36.77\%
$$

再加入 BT：

$$
39.18\%.
$$

作者报告 GR、CAA、BT 和 KGCE 都有明显贡献。

不过这个对你来说主要是“论文设计参考”，不建议直接移植，因为 GR 明显涉及训练。

---

# 十三、BT 模块：简单，但非常值得你从反面借鉴

论文最后做了：

$$
\max_c P_c(x)<\tau
\Rightarrow background.
$$

也就是说如果所有 foreground class response 都低于固定阈值，就设为 background。

最佳：

$$
\tau=0.6
$$

作者发现 0.1–0.3 较稳定，0.6 达到峰值，再继续增加则下降。

但这个模块恰恰暴露了明显问题。

论文自己承认：

固定 threshold 会：

**压掉边界低置信度像素；**

同时在 FLAIR#1 这种 long-tail dataset 上：

**rare class 容易被误判成 background。** 

---

# 十四、这里可以直接映射到你现在的 risk gating

这其实是一个很好的对比。

KG-OVRSeg：

$$
\text{fixed confidence threshold}.
$$

你应该强调：

$$
\text{response-conditioned / risk-aware decision}.
$$

也就是说不是简单：

> confidence low → background

而是综合：

```text
prompt reliability
class competition
response consistency
presence evidence
risk
```

然后再决定抑制还是保留。

从论文设计角度讲：

**你不应该向它的 BT 靠拢，反而应该把它当成固定 threshold 局限性的一个相关工作例子。**

---

# 十五、实验设计上，这篇论文非常值得借鉴

作者不是只做了最终 mIoU 表，而是建立了一整套“证明语义表示真的变好了”的证据链。

这个特别值得你学。

论文 Fig.11 做 cosine similarity matrix，证明加入 KGCE 后，同组类别比如：

`barren / bareland / bare soil`

以及：

`grassland / grass cover`

变得更接近。

Fig.12 又做 t-SNE：

未使用 KGCE 时，同一 semantic group 非常散；

使用 KGCE 后：

> intra-class compactness ↑
> inter-class separability ↑



Fig.14 再直接做 text→image response maps，证明不同 class word 激活完全不同。

最后 Fig.13 还把 KGCE 插到不同 OVSS 方法中，平均：

$$
+4.6\% \text{ mean mF1}
$$

$$
+3.5\% \text{ mean mIoU}
$$

证明模块不是只对自己的 backbone 有效。

---

# 十六、这套“证据链”非常适合你的 RPS-SAM3

你的论文以后最好不要只证明：

> DAG prompt → mIoU +2.x

而应该证明完整的因果链：

```text
Dataset class name
        ↓
semantic / model-response ambiguity
        ↓
different candidate prompts
produce different SAM3 responses
        ↓
unsafe expansion increases cross-class confusion
        ↓
DAG search removes unstable / confusing prompts
        ↓
PromptBank becomes more reliable
        ↓
mIoU increases
```

我甚至认为，你应该有一张类似论文 Fig.14、但更加符合 SAM3 的实验图。

例如每行一个 class：

```text
GT / image

Native
Synonym A
Synonym B
Hyponym A
Context Prompt
Unsafe Prompt
Selected Prompt Set
```

下面分别画 SAM3 response。

这会非常有杀伤力。

因为它不是抽象地说：

> prompt matters。

而是直接告诉 reviewer：

> **同一个 frozen SAM3，同一张图，只改几个词，响应空间发生巨大变化。**

这正是你的论文核心故事。

---

# 十七、这篇论文和你的工作到底重不重合？

这一点非常重要。

我认为：

## 有明显的 motivation overlap，但 method overlap 并不严重。

可以这样区分：

|                   | KG-OVRSeg                 | 你的方向                                    |
| ----------------- | ------------------------- | --------------------------------------- |
| 核心问题              | semantic ambiguity        | prompt reliability / semantic ambiguity |
| 基础模型              | CLIP-based OVSS           | SAM3 / SegEarth-OV3                     |
| 类别扩展              | Knowledge Graph           | DAG candidate search                    |
| 候选依据              | ontology relation         | semantic + SAM3 response                |
| 选择机制              | 基本没有，直接聚合                 | reliability / safety pruning            |
| representation    | node embedding average    | selected prompt responses / PromptBank  |
| target evidence   | 不用于 prompt 选择             | frozen SAM3 response evidence           |
| training          | **需要训练 decoder**          | **training-free**                       |
| GT                | OEM supervised training   | no masks / GT-free search               |
| online adaptation | 无                         | 无                                       |
| class conflict    | ontology global filtering | cross-class response safety             |
| robustness        | 一般 cross-domain           | 可专门研究 corruption robustness             |

所以：

### 不会直接否定你的创新。

但它会让下面这种表述变得危险：

> “Previous RS-OVSS methods ignore semantic ambiguity of class names.”

因为 KG-OVRSeg 已经明确研究了这个问题。

你应该改成更准确的 gap：

> Existing work addresses class-level ambiguity through manually curated structured knowledge and learned semantic decoders, but does not determine whether a candidate phrase is actually reliable for the frozen segmentation model itself.

或者更加凝练：

> **knowledge-consistent ≠ model-response-reliable.**

我认为这句话甚至可以成为你论文非常好的理论分界点。

---

# 十八、两篇工作的真正哲学差异

KG-OVRSeg 假设：

$$
\text{good semantic knowledge}
\Rightarrow
\text{good class representation}.
$$

你的工作更应该强调：

$$
\text{linguistically valid prompt}
\not\Rightarrow
\text{reliable visual response}.
$$

这是很重要的区别。

例如：

```text
road
arterial road
minor road
street network
traffic lane
road pavement
```

从知识图谱角度它们都合理。

但是 frozen SAM3 可能：

```text
road             → strong
road pavement    → strong
traffic lane     → unstable
street network   → over-segmentation
minor road       → weak
```

所以：

### KG 只能说明 candidate “语义上合理”。

不能证明：

### candidate “对当前 foundation model 有效”。

这正好是你 RPS / DAG search 可以占据的位置。

---

# 十九、我最建议你借鉴的技术路线

如果让我把这篇论文真正融合进你现在的工作，我不会做：

```text
LCCKG
→ average text embedding
→ SAM3
```

而会做：

$$
\boxed{
\text{Knowledge-Constrained Response-Guided DAG Search}
}
$$

整体变成：

```text
Immutable Class Anchor
            │
            ↓
  Semantic Relation Graph
 synonym / hyponym / attribute / context
            │
            ↓
 Dataset-Level Global Filtering
 hierarchy overlap / cross-class collision
            │
            ↓
 Frozen SAM3 Response Cache
            │
            ↓
 Response Reliability Evaluation
 stability / confidence / discrimination
            │
            ↓
 Safety + Pareto + Upper-bound Pruning
            │
            ↓
      Frozen PromptBank
```

这里：

**KG-OVRSeg 提供 semantic topology。**

而你的 RPS：

**提供 model-response evidence。**

两个结合之后的理论逻辑会比“LLM 随机扩词 → search”扎实很多。

---

# 二十、甚至可以给 DAG 增加一个 hierarchy safety constraint

现在可以考虑一个 candidate \(p\) 对 class \(c\)：

$$
Q(p,c)
=
\alpha S
+\beta A
+\gamma D
-\lambda C
-\mu H
$$

其中不用拘泥于这个具体公式，但概念上：

\(S\)：response stability
\(A\)：response strength / agreement
\(D\)：class discrimination
\(C\)：cross-class confusion
\(H\)：hierarchy violation

所谓 hierarchy violation，例如：

如果：

```text
agriculture
├── vineyard
└── plowed land
```

而当前 dataset 同时把：

```text
agriculture
vineyard
plowed land
```

作为三个独立 class，

那么：

`vineyard`

就不能再轻易进入：

`agriculture`

PromptBank。

这就是 KG-OVRSeg global filtering 的思想，但转换成：

> **DAG search safety rule。**

而且不需要任何训练。

---

# 二十一、这篇论文实验上还有一个很聪明的点：cache

KGCE 第一次需要编码大量 subgraph nodes，所以第一次 inference 比较慢：

w/o KGCE：

$$
937.351ms
$$

w KGCE：

$$
2624.143ms
$$

但 embedding cache 之后：

$$
32.665ms
\rightarrow33.890ms.
$$

也就是说稳定推理阶段的额外成本非常小。

作者明确采用 class encoding cache，同一个类别列表只编码一次。

这个实验设计你也应该借鉴。

你现在本身就是：

> offline PromptBank construction → frozen online inference

因此以后计算开销最好明确拆成：

```text
Offline prompt search cost
One-time PromptBank encoding cost
Online stable inference latency
Peak GPU memory
FPS
```

这样 reviewer 就很难说：

> “你的 prompt search 太慢，所以方法不实用。”

因为你可以明确：

> 搜索只进行一次，不属于 test-time inference。

---

# 二十二、但这篇论文也有几个明显弱点，你不要照搬

这里我认为需要客观看。

### 第一，所谓强 cross-domain performance 有训练协议因素

它用：

> OpenEarthMap train

训练模型，再在七个数据集上 test，其中甚至包括：

> OpenEarthMap test。



论文自己也承认，这导致一些训练型方法在 OEM 上特别强，而 SegEarth-OV 因为没有进行相同训练，在这个比较里比较吃亏。

所以它和你的：

> completely frozen / no target training

不是同一个 protocol。

这一点不要被 Table 2 的巨大增益吓到。

---

### 第二，Knowledge Graph 强依赖人工先验

LCCKG 并不是凭空来的。

它：

> 结合 127 个 RS segmentation datasets，
> 并由 domain experts 标注层级关系。



因此它的 open vocabulary 在一定程度上依赖：

**知识图谱的 coverage。**

作者自己也承认：

如果遇到 out-of-graph class：

> performance tends to revert to baseline。



这正是你 DAG/LLM dynamic candidate generation 可以超过它的地方。

---

### 第三，average pooling 太粗糙

这是我认为方法上最明显的弱点之一。

它完全忽略：

```text
relation type
node reliability
node visual response
node importance
cross-class confusion
```

最后：

$$
AvgPool
$$

全部吃掉。

这和论文 Fig.14 自己观察到的：

> 不同词 response 差异极大

实际上存在一定张力。

而你的 response-aware selection 正好可以解决这个问题。

---

### 第四，fixed background threshold 不够稳健

作者自己已经证明：

长尾 class 和 object boundary 会受到影响。

而且论文展示了 \(\tau=0.6\) 最优，但从正文并不能十分清楚地确认：

> 这个阈值是否严格只在独立 validation protocol 上选定。

因此如果做更严格的 training-free benchmark，threshold selection protocol 一定要写得比这篇论文更加严格。

---

# 二十三、如果只选“最值得借鉴的五件事”

| 优先级   | 借鉴内容                                       | 对你价值                            |
| ----- | ------------------------------------------ | ------------------------------- |
| **S** | dataset-level global semantic filtering    | 直接强化 Cross-Class Safety         |
| **S** | subclass / contain 比 synonym 更重要           | 改造 DAG candidate taxonomy       |
| **S** | class-word response map 分析                 | 非常适合增强 RPS-SAM3 论文证据            |
| **A** | KG proposal + model response validation    | 很可能形成新的高创新扩展                    |
| **A** | offline cache / stable inference reporting | 强化 training-free 方法实用性          |
| A     | embedding intra/inter-class analysis       | 增强机制解释                          |
| B     | image-Q / text-KV 思想                       | 可转化为 class-response competition |
| C     | fixed BT                                   | 不建议照搬                           |
| C     | average pooling all KG nodes               | 不建议照搬                           |

---

# 二十四、对你当前工作的影响：我认为是利大于弊

这篇文章并不会让我认为你现在的 RPS-SAM3/DAG Prompt Search 没有价值。

恰恰相反。

它帮助你把理论故事补完整了：

### KG-OVRSeg 已经证明第一层：

$$
\boxed{\text{Dataset label semantics are ambiguous}}
$$

而你真正应该证明第二层：

$$
\boxed{
\text{Semantically valid prompts are not necessarily
model-response reliable}
}
$$

然后进一步提出：

$$
\boxed{
\text{Prompt reliability must be validated against
the frozen foundation model itself}
}
$$

于是相关工作的关系就非常清楚：

```text
TACOSS
    ↓
How should land-cover classes be described?

KG-OVRSeg
    ↓
How can structured semantic knowledge enrich class representations?

RPS / DAG-SAM3
    ↓
Which semantically valid prompts are actually safe and reliable
for a frozen segmentation foundation model?
```

这个定位我认为比简单写：

> “we propose better prompts”

要强得多。

---

# 二十五、我认为最值得你立即做的实验

如果下一步是为了完善你当前那篇 **training-free SAM3 / DAG PromptBank** 工作，我会优先增加一组 **Hierarchy-Aware DAG** 实验，而不是去复现它整个 KG-OVRSeg。

可以直接做成：

| Variant        | Candidate space   | Global filtering | Frozen response screening |
| -------------- | ----------------- | ---------------: | ------------------------: |
| Native         | native label      |                × |                         × |
| Synonym        | synonym           |                × |                         × |
| Hierarchy      | synonym + hyponym |                × |                         × |
| Hierarchy-Safe | synonym + hyponym |                ✓ |                         × |
| Response-RPS   | existing DAG      |                ✓ |                         ✓ |
| **Full**       | semantic DAG      |                ✓ |                         ✓ |

然后重点看：

```text
mIoU
per-class IoU
cross-class confusion
PromptBank size
unsafe prompt rejection rate
first/offline cost
stable FPS
```

如果最终出现类似：

```text
Synonym only          +0.x
Hierarchy             +1.x
Hierarchy + safety    +2.x
Hierarchy + response  +3.x
```

那么你的 DAG 就不再像“prompt engineering”。

它会变成：

> **一个有明确 semantic topology、dataset-level competition 和 frozen-model evidence 的 structured search framework。**

这才是这篇 ISPRS JPRS 论文对你最大的价值。

**一句话总结：这篇论文最该借鉴的不是 KGCE 的“平均 embedding”，而是它把遥感类别词从孤立 label 提升成“具有层级、包含、同义和排他关系的结构化语义空间”；你则完全可以再向前走一步，把这个结构化语义空间与 frozen SAM3 的真实 response evidence 结合，形成“Knowledge-Constrained + Response-Validated DAG Prompt Search”。** 这会比单纯复用 KG 更符合你现在 training-free SAM3 路线，也更容易与这篇论文形成清晰的创新边界。
