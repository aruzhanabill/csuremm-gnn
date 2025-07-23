// src/App.js
import React from 'react';
import Hero from './components/Hero';
import Section from './components/Section';
import InteractiveDiagram from './components/InteractiveDiagram';
import Footer from './components/Footer';
import { InlineMath } from 'react-katex';
import { BlockMath } from 'react-katex';

 
function App() {
  return (
    <div>
        <Hero />
        <Section
        title=""
        content={
            <>
            <p>
                The nonlinear and volatile nature of financial data makes accurate stock price prediction a persistent
                challenge. Despite effectively modeling inter-stock dependencies, existing Graph Neural Networks (GNNs)
                fail to capture the structural supply chain changes induced by macroeconomic shocks.
            </p>
            <p>
                To address this challenge, we propose <strong>OmniGNN</strong>, an attention-based multi-relational dynamic GNN that
                integrates macroeconomic context via heterogeneous node and edge types for robust message propagation.
                Central to OmniGNN is a sector node acting as a global intermediary, enabling rapid shock propagation
                across the graph without relying on long-range multi-hop diffusion.
            </p>
            <p>
                The model leverages Graph Attention Networks (GATs) to weigh neighbor contributions and employs Transformers
                to capture temporal dynamics across multiplex relations. Experiments show that OmniGNN outperforms existing
                stock prediction models on public datasets, particularly demonstrating strong robustness during the COVID-19
                period.
            </p>
            </>
        }
        />
        <Section
        title="Graph neural networks and their current limitations"
        content={
            <>
            <p>
                Recent fusion models that pair Graph Convolutional Networks (GCNs) with temporal architectures like Long
                Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs) have set the benchmark for node classification 
                tasks in financial domains, due to their ability to model nonlinear interactions across time. Within 
                the equity market, GCNs encode the topological structure of stock interactions, while Gated Recurrent Neural 
                Networks (RNNs) capture the temporal dependencies.
            </p>
            <p>
                However, such models face several limitations. First, although the sequential processing mechanism of RNNs 
                is well-suited for capturing long-range temporal dependencies, these models encounter memory bottlenecks as 
                complexity increases, limiting their capacity to integrate information across the full temporal history. Second, 
                message-passing GCNs that rely solely on one-hop neighborhood aggregation fail to distinguish non-isomorphic 
                graphs with similar local structures. Additionally, the standard GCN aggregation scheme applies uniform averaging 
                rather than assigning learnable importance weights to neighboring nodes. Third, many GNNs face the challenge of 
                oversmoothing, which is the tendency for multiple iterations of message-passing to generate homogeneous node 
                embeddings, making it difficult for the GNN to learn longer-term dependencies in the graph. While vector concatenations 
                and skip connections are utilized to preserve previous node-level information during the update step, these strategies 
                are signal-based, rather than structural, and therefore insufficient for preserving the full complexity of the evolving 
                network topology.
            </p>
            <p>
                To address these challenges, we introduce OmniGNN, a novel GAT-Transformer-ALiBi architecture, encapsulating three core 
                contributions: (i) Attention-weighting for metapaths: We define new metapath structures that leverage self-attention 
                to prioritize informative sequences of neighboring nodes. (ii) Temporal encoding via transformer: Unlike RNNs, our 
                temporal module uses a Transformer encoder with Attention with Linear Biases (ALiBi), allowing each timestep to attend 
                to all others in parallel, with inductive bias toward recent observations. (iii) Global node for multi-hop message 
                propagation: We propose the inclusion of an industry node, a global node connected to all stocks in that industry at once 
                via multi-relational edges that reflect supply-chain, regulatory, and sectoral ties.
            </p>
            </>
        }
        />
        <Section
        title="OmniGNN architecture"
        content={
            <>
            <p>
                The OmniGNN model architecture is comprised of three layers: 1) Structural Layer, which transforms nodes into their 
                weighted metapath representations; 2) Temporal Layer, which learns node representations across time windows, and 3) 
                Prediction Layer, which computes the next day excess return for each stock node on trading day <InlineMath math="t"/>. 
                The model architecture is visualized below:
            </p>
            <img
                src="/Omni-GNN Model Diagram.png"
                alt="OmniGNN architecture diagram"
                style={{
                    width: '90%',
                    borderRadius: '12px',
                    marginTop: '1rem',
                    display: 'block',
                    marginLeft: 'auto',
                    marginRight: 'auto'
                }}
            />
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Constructing the graph</h3>
            <p>
                Let <InlineMath math="\mathcal{G} = \{\mathcal{G}^{(t)}\}_{t=1}^T"/>, be a sequence of discrete, multi-relational graphs 
                indexed by time <InlineMath math="t"/>, from trading day <InlineMath math="1"/> to <InlineMath math="T"/>. Each graph 
                snapshot is defined as <InlineMath math="\mathcal{G}^{(t)} = \{\mathcal{V}^{(t)}, \mathcal{A}^{(t)}, \mathcal{E}^{(t)}\}"/>,
                where:
            </p>
            <ul>
                <li><InlineMath math="\mathcal{V}^{(t)}"/> is the fixed set of <InlineMath math="N"/> nodes at time <InlineMath math="t"/>.
                Each node <InlineMath math="v_i \in \mathcal{V}^{(t)}"/> is associated with a feature vector <InlineMath math="x_i^{(t)}
                \in\mathbb{R}^F"/> and a scalar label <InlineMath math="y_i^{(t)}\in\mathbb{R}"/>, where <InlineMath math="F=16"/> is the 
                node feature dimension.</li>
                <li><InlineMath math="\mathcal{A}^{(t)} \in \{0, 1\}^{|\mathcal{R}| \times N \times N}"/> is a multi-relational binary 
                adjacency tensor encoding edge existence under relation types <InlineMath math="\mathcal{R}=\{\mathcal{SS}, \mathcal{SI}\}"/>
                .</li>
                <li><InlineMath math="\mathcal{E}^{(t)} \in \mathbb{R}^{|\mathcal{R}| \times N \times N \times E}"/> stores multiple edge 
                attributes for each relation type, where  <InlineMath math="E=2"/> is the edge feature dimension.</li>
            </ul>
            <p> 
                Concretely, we represent a stock as <InlineMath math="\mathcal{S}"/> and industry as <InlineMath math="\mathcal{I}"/>, 
                and define their corresponding edges <InlineMath math="\mathcal{E}_{\mathcal{SS}}"/> and <InlineMath math="\mathcal{E}
                _{\mathcal{SI}}"/> using various inter-market relations between stocks and their industries.
            </p>
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Embedding the graph</h3>
            <h4>Metapath construction</h4>
            <p>
                As different node and edge features distinctly impact stock nodes, we define a set of two metapaths 
                <InlineMath math="\mathcal{P} = \{\mathcal{SS}, \mathcal{SIS}\}"/>. Metapaths act as paths through the network schema, 
                revealing higher-level connections between stock nodes. For example, to calculate the adjacency of the <InlineMath math="\mathcal{SIS}"/>
                metapath, we perform matrix multiplication to chain together intermediate edges: 
            </p>
            <BlockMath math="A_{\mathcal{SIS}} = A_{\mathcal{SI}} \cdot A_{\mathcal{IS}} = A_{\mathcal{SI}} \cdot A_{\mathcal{SI}}^{\top}, 
            \qquad A_{\mathcal{SIS}}\in\{0,1\}^{N\times N}."/>
            <p> 
                Edge features for <InlineMath math="\mathcal{SIS}"/> are computed as the simple average of intermediate edge attributes along the path:
            </p>
            <BlockMath math="E_{\mathcal{SIS}}[i,j] = \frac{1}{2}\left(E_{\mathcal{SI}}[i,k] + E_{\mathcal{SI}}[j,k]\right),"/>
            <p>
                where <InlineMath math="k"/> indexes the shared industry node.
            </p>
            <h4>Graph Attention Mechanism</h4>
            <p>
                For a set of node features <InlineMath math="h = {\{\vec{h_1}, \vec{h_2}, \vec{h_3}, \cdots, \vec{h_n}}\} h_i \in \mathbb{R}^F"/>,
                where <InlineMath math="N"/> and <InlineMath math="F"/> is the number of features, our goal is to obtain a context-aware node embedding 
                that incorporates the most relevant information from neighboring nodes. To enable full pairwise attention, we adopt the architectural 
                framework proposed by Graph Attention Networks.
            </p>
            <img
                src="/Attention Diagram.png"
                alt="Attention Diagram"
                style={{
                    width: '75%',
                    borderRadius: '12px',
                    marginTop: '1rem',
                    display: 'block',
                    marginLeft: 'auto',
                    marginRight: 'auto'
                }}
            />
            <p> 
                In a single graph attention layer setup, we begin by applying a shared linear transformation, represented by the weight matrix <InlineMath math=" \mathbf{W}"/>, 
                to each node feature vector as well as to each edge attribute associated with the graph. Next, we 
                compute self-attention scores using a learnable attention mechanism <InlineMath math="a : \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow 
                \mathbb{R}"/>. For each node pair <InlineMath math="(i,j)"/>, we compute the attention coefficient, representing the importance of the features 
                of node <InlineMath math="\mathbf{j}"/> for node <InlineMath math="\mathbf{i}"/>, as follows: 
            </p>
            <BlockMath math="\beta_{ij} = \vec{a}^T(\mathbf{W} {h_i} || \mathbf{W}{h_j} || \mathbf{W} {{edge\_attr}_{ij}})."/>
            <p> 
                After enabling nonlinearity through LeakyReLU, the attention coefficient is  normalized across node features via softmax activation 
                (<InlineMath math="\alpha = 0.2"/>):
            </p>
            <BlockMath math="a_{ij} = \frac{exp(LeakyReLU(\beta_{ij}))}{\sum_{k \in \mathcal{N_i}} exp(LeakyReLU(\beta_{ij}))},"/>
            <p>where <InlineMath math="\mathcal{N}_i"/> represents the neighborhood of the node, determined by the adjacency matrix.</p>
            <p>
                Finally, the resulting node embeddings <InlineMath math="\mathbf{h'} = {\{\vec{h'_1}, \vec{h'_2}, \vec{h'_3}..., \vec{h'_n}}\}, h_i \in 
                \mathbb{R}^{F'}"/> are updated by aggregating across <InlineMath math="H"/> attention heads via average pooling:
                <BlockMath math="\mathbf{h}_i' = \frac{1}{H}\sum_{k=1}^H\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j \in \mathbb{R}^{H \cdot F}."/>.
            </p> 
            <p>
                This sequence is repeated for each time slice <InlineMath math="t-\delta_t,\cdots,t"/>, producing a temporal sequence 
                <InlineMath math="\mathbf{H}_{v,t-\delta_t:t} \in \mathbb{R}^{\delta_t\times d_h}"/>, for each node <InlineMath math="v"/>.
            </p> 
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Sequencing the graph</h3>
            <p>
                While the structural layer represents nodes in such a way that encodes information about their semantic relationships 
                within the corporate network on a certain day <InlineMath math="t"/>, these relationships are constantly evolving. For example, stock node 
                features such as price, volatility, and news sentiment fluctuate daily as the market changes, while institutional 
                shareholders will enter or exit positions over time, altering the network of ownership ties. Therefore, it is essential 
                to extract the temporal element of node and edge features from the graph snapshot sequence. To accomplish this task, we 
                employ the Transformer mechanism with ALiBi. The Transformer has made watershed progress in 
                the realm of large language models with its ability to integrate context into token embeddings. Its general structure to 
                learn sequence-to-sequence data is repurposed here for temporal modeling of graph snapshots. 
            </p>
            <img
                src="/Transformer Diagram.png"
                alt="Transformer Diagram"
                style={{
                    width: '75%',
                    borderRadius: '12px',
                    marginTop: '1rem',
                    display: 'block',
                    marginLeft: 'auto',
                    marginRight: 'auto'
                }}
            />
            <p>
                The sequence <InlineMath math="\mathbf{H}_{v,t-\delta_t:t}"/> is passed through a Transformer with ALiBi for 
                temporal modeling. The temporal attention computes:
            </p>
            <BlockMath math="\mathbf{Q} = \mathbf{W}_\mathbf{Q}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}}, \nonumber \qquad
                            \mathbf{K} = \mathbf{W}_\mathbf{K}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}}, \nonumber \qquad
                            \mathbf{V} = \mathbf{W}_\mathbf{V}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}},"/>
            <BlockMath math="\mathbf{Z} = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}} + m \cdot \mathbf{P} + \mathbf{M}\right)\mathbf{V}."/>
            <p>
                where <InlineMath math="\mathbf{W_{\mathbf{Q}}}"/>, <InlineMath math="\mathbf{W_{\mathbf{K}}}"/>, and <InlineMath math="\mathbf{W_{\mathbf{V}}}"/> 
                are trainable weight matrices, <InlineMath math="m\cdot \mathbf{P}"/> is the linear recency bias term added to the attention scores with 
                <InlineMath math="m"/> controlling the strength of the bias, and <InlineMath math="\mathbf{M}"/> acts as a causal mask, ensuring the model attends 
                only to the past and present time steps. 
            </p>
            <p>
                The temporal layer outputs node embeddings <InlineMath math="\mathbf{z}_i^{(t)}"/>, the final representation for the stock nodes 
                <InlineMath math="v"/> on trading day <InlineMath math="t"/>.
            </p>
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Prediction</h3>
            <p>
                Each stock node has a stock-specific linear predictor:
            </p>
            <BlockMath math="\hat{y}_{vt} = \mathbf{W}_1\mathbf{z}_{vt}+\mathbf{b}_1,"/>
            <p>
                Ground truth is defined to be the <strong>excess return</strong> over the benchmark <InlineMath math="SPX"/> index:
            </p>
            <BlockMath math="y_{it} = \frac{p_{i,t+1}\;-\;p_{i,t}}{p_{i,t}} - \frac{SPX_{i,t+1}\;-\;SPX_{i,t}}{SPX_{i,t}}."/>
            </>
        }
        />
        <Section
        title="Experiments and Results"
        content={
            <>
            <h3>Experimental Design</h3>
            <p>
                We adopt a rolling window backtesting strategy to evaluate the model's robustness across the varying market 
                conditions represented in the period from January 5th, 2019 through the end of 2022. The full historical dataset 
                is divided into overlapping windows that segment 6 months for training, 2 months for validation, and 2 months for 
                testing. The Information Coefficient, Information Ratio, Cumulative Return, and Precision@K are computed for each 
                cycle, and the window advances by the length of the testing period. Model performance on the full historical 
                dataset is evaluated by averaging the values of each window's model's performance metrics.
            </p>
            <h3>Results</h3>
            <img
                src="/Results.png"
                alt="Results"
                style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }}
            />
            <p>
               The Transformer model is a time-series method that captures temporal dependencies in node features, but doesn't 
               depend on graph structure. The GAT model is insensitive to temporal dynamics, but it effectively captures local 
               relational patterns in the knowledge graph. The Transformer model outperforms the GAT in all metrics, which suggests 
               that the temporal aspect of our 10-stock corporate network has the greatest impact on improving the accuracy of 
               predicting excess returns. OmniGNN achieves superior predictive performance across all metrics likely because it 
               combines both temporal and structural reasoning to learn how the complex relationships between stocks and industries 
               evolve over time.
            </p>
            <h3>Case Study</h3>
            <p>
               To test the impact of the global node on model performance, we perform an ablation experiment by removing the industry 
               node <InlineMath math="\mathcal{I}"/>, the <InlineMath math="\mathcal{E}_{\mathcal{SI}}"/> edge, and the corresponding 
               <InlineMath math="\mathcal{SIS}"/> metapath. The backtest is performed on the COVID-19 period from March 4, 2020 to 
               May 4, 2020. 
            </p>
            <img
                src="/Ablation Case Study Results.png"
                alt="Case Study Results"
                style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }}
            />
            <p>
                Inclusion of the <InlineMath math="\mathcal{SIS}"/> metapath contributes to significant gains in IC, IR, and Precision@K 
                by reinforcing industry-level relationships among stocks. The global node serves as a centralized message-passing hub, 
                enabling efficient information propagation across the graph. This design mitigates gradient explosion during training and 
                preserves graph topology under stress, resulting in more stable optimization. 
            </p>
            </>
        }
        />
        <Section
        title="Conclusion"
        content={
            <>
            <p>
                In this work, we introduce the idea of a global node to ease message-passing by propagating industry-level shocks explicitly 
                to every other node in the graph, instead of relying on iterative message-passing between distant nodes that leads to 
                oversmoothing of node representations. In this paper, the knowledge graph that OMNI-GNN intakes consists of 10 stock nodes 
                and 1 industry (global) node. This size limits our evaluation of how effective the global node is at increasing message-passing 
                efficiency, because the stock nodes in our graph are always at most 2 hops away from each other when the global node is ablated. 
                Future work will focus on extending our experiments to incorporate far more than just 10 stocks, but entire corporate networks, 
                such as the entire Information Technology industry or S&P 500.
            </p>
            </>
        }
        />
        <Footer />
    </div>
  );
}

export default App;