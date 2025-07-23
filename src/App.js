// src/App.js
import React from 'react';
import Hero from './components/Hero';
import Section from './components/Section';
import InteractiveDiagram from './components/InteractiveDiagram';
import Footer from './components/Footer';
import { InlineMath } from 'react-katex';
import { BlockMath } from 'react-katex';
import Ablation from './Ablation-Results.jpg';
import OmniGNN from './omni-gnn-model-diagram.png';
import AttentionDiagram from './Attention-Diagram.png';
import BaselineResults from './Baseline-Results.png';
import BaselineResultsGraph from './Baseline-Results-Graph.png';
import TransformerDiagram from './Transformer-Diagram.png'; 



function App() {
  return (
    <div>
        <Hero />
        <Section
        title=""
        content={
            <>
            <p>
                The nonlinear and chaotic nature of financial data makes accurate stock price prediction a persistent
                challenge. While Graph Neural Networks (GNN) effectively model inter-stock dependencies, existing models
                fail to efficiently propagate messages during macroeconomic shocks. 
            </p>
            <p>
                To address this challenge, we propose <strong>OmniGNN</strong>, an attention-based multi-relational dynamic GNN that
                integrates macroeconomic context via heterogeneous node and edge types for robust message propagation.
                Central to OmniGNN is a sector node acting as a global intermediary, enabling rapid shock propagation
                across the graph without relying on long-range multi-hop diffusion.
            </p>
            <p>
                The model leverages Graph Attention Networks (GAT) to weigh neighbor contributions and employs Transformers
                to capture temporal dynamics across multiplex relations. Experiments show that OmniGNN outperforms existing
                stock prediction models on public datasets, particularly demonstrating strong robustness during the COVID-19
                period.
            </p>
            <p>
                The code and models of OmniGNN are made publicly available at https://github.com/amberhli/CSUREMM-3
            </p>
            </>
        }
        />

        <Section
        title="Graph neural networks and their current limitations"
        content={
            <>
            <p>
                In recent years, Graph Neural Networks (GNNs) has been established as the standard for prediction tasks on graph 
                datasets at the graph, edge, and node-level with significant applications in areas such as molecular modeling, 
                transportation networks, and recommendation systems. Particularly, in financial domains, recent 
                fusion models that pair Graph Convolutional Networks (GCNs) with temporal Recurrent Neural Networks (RNNs) like 
                Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs) have set the benchmark for dynamic node-level 
                prediction tasks using structures like stock correlation or company knowledge graphs. 
            </p>
            <p>
                These models, however, face several limitations. First, although RNNs are well-suited for capturing long-range 
                temporal dependencies, their sequential nature causes memory bottlenecks with the increase in complexity, limiting 
                their capacity to integrate information across the full temporal history . Second, message-passing GCNs that rely 
                solely on one-hop neighborhood aggregation fail to distinguish nonisomorphic graphs with similar local structures.
                Additionally, the standard GCN aggregation scheme applies uniform averaging to incoming messages, which fails to 
                acknowledge the varying relevance of such messages. Third, many GNNs face the challenge of oversmoothing, generating 
                homogeneous node embeddings after multiple rounds of message-passing, making it difficult for the GNN to learn 
                longer-term dependencies in the graph. While skip connections are commonly utilized to preserve previous node-level 
                information during the update step, this strategy not only increase parameter count and model depth, but leave the graph 
                susceptible still to oversmoothing.
            </p>
            <p>
                To address these challenges, we introduce <strong>OmniGNN</strong>, a novel GAT-Transformer architecture, including three core 
                contributions.
            </p>
            <p>
                First, we propose <em>Metapath Attention-Weighting</em> for relational structure modeling. Specifically, we define a set of 
                metapaths that originate and terminate at stock nodes while traversing intermediate nodes of varying types (e.g., industries 
                or regulatory entities). These metapaths capture higher-order semantic relationships that extend beyond stock-to-stock 
                dependencies—for example,  structural correlations arising from shared industry affiliations. By assigning attention weights 
                to these metapaths during message aggregation, OmniGNN can effectively filter noise from less relevant paths and improve the 
                expressiveness of the learned embeddings. 
            </p>
            <p>
                Second, we propose <em>Temporal Encoding via Transformer</em> to capture nuanced, longer-range dependencies in the evolving 
                network. Unlike RNNs, Transformers eliminate sequential recurrence, allowing each time step to attend to all others 
                simultaneously, which significantly reduces training time by enabling full parallelization across sequence elements. 
                To the Transformer, we apply Attention with Linear Biases (ALiBi), which applies linear positional bias to the attention 
                score computation. ALiBi preserves the model's ability to generalize to longer sequences without having to learn 
                positional embeddings, while placing inductive bias toward recent observations in the graph network.
            </p>
            <p>
                Third, we propose a novel <em>"Global" Node</em> for a more robust network topology. We design an industry node, or a 
                global node connected to all stocks in that industry at once via multi-relational edges that reflect supply-chain, regulatory, 
                and sectoral ties. This global node creates a star topology overlay such that any two nodes in the graph can communicate to 
                each other in 2 hops, shortening the message-passing path. Equipped with this virtual connection, our model can at once retain 
                local information while capturing distant interactions without relying on as many GNN layers, effectively mitigating the 
                oversmoothing challenge.
            </p>
            </>
        }

        />
        <Section
        title="OmniGNN model architecture"
        content={
            <>
            <p>
                OmniGNN consists of three key layers: 1) Structural Layer — encodes nodes using weighted metapath representations. 2) Temporal 
                Layer — learns dynamic node representations across time windows. 3) Prediction Layer — outputs the next day excess return for 
                each stock node on trading day <InlineMath math="t"/>. The model architecture is visualized in Figure 1.
            </p>
            <img
                src={OmniGNN}
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
            <h3>Layer 1: Embedding the Graphs</h3>
            <h4>Metapath construction</h4>
            <p>
                As different node and edge features distinctly impact stock nodes, we define a set of two metapaths 
                <InlineMath math="\mathcal{P} = \{\mathcal{SS}, \mathcal{SIS}\}"/>. Metapaths act as paths through the network schema, 
                revealing higher-level connections between stock nodes. For example, to calculate the adjacency of the <InlineMath 
                math="\mathcal{SIS}"/> metapath, we perform matrix multiplication to chain together intermediate edges: 
            </p>
            <BlockMath math="A_{\mathcal{SIS}} = A_{\mathcal{SI}} \cdot A_{\mathcal{IS}} = A_{\mathcal{SI}} \cdot A_{\mathcal{SI}}^{\top}, 
            \qquad A_{\mathcal{SIS}}\in\{0,1\}^{N\times N}."/>
            <p> 
                Edge features for <InlineMath math="\mathcal{SIS}"/> are computed as the average of intermediate edge attributes along the path:
            </p>
            <BlockMath math="E_{\mathcal{SIS}}[i,j] = \frac{1}{2}\left(E_{\mathcal{SI}}[i,k] + E_{\mathcal{SI}}[j,k]\right),"/>
            <p>
                where <InlineMath math="k"/> indexes the shared industry node. Diagonal entries for both 
                <InlineMath math="A_{\mathcal{SIS}}"/> and <InlineMath math="E_{\mathcal{SIS}}"/> are set to <InlineMath math="1"/> to include 
                self-loops. Note that <InlineMath math="A_{\mathcal{SS}}"/> and <InlineMath math="E_{\mathcal{SS}}"/> are simply stock-stock edges.
            </p>

            <h4>Graph Attention Mechanism</h4>
            <p>
                For a set of node features <InlineMath math="\mathbf{h} = {\{\mathbf{h}_1, \mathbf{h}_2, \cdots, \mathbf{h}_N}\}, \mathbf{h}_i \in \mathbb{R}^F"/>,
                where <InlineMath math="N"/> and <InlineMath math="F"/> is the number of features, our goal is to obtain a context-aware node embedding 
                that incorporates the most relevant information from neighboring nodes. To enable full pairwise attention, we adopt the architectural 
                framework proposed by Graph Attention Networks.
            </p>
            <img
                src={AttentionDiagram}
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
                In a single graph attention layer setup, we first apply a shared linear transformation, represented by the weight matrix <InlineMath math=" \mathbf{W}"/>, 
                to both the node feature vectors and the edge attributes of the graph. For each node pair <InlineMath math="(v_i,v_j)"/>, the attention coefficient <InlineMath math="\beta_{ij}"/>, 
                measuring the importance of <InlineMath math="v_j"/>'s features to <InlineMath math="v_i"/>'s, is computed as follows: 
            </p>
            <BlockMath math="\beta_{ij} = a^T(\mathbf{W} {\mathbf{h}_i} || \mathbf{W}{\mathbf{h}_j} || \mathbf{W} {\mathbf{e}_{ij}}),"/>
            <p>
                where <InlineMath math="a"/> is a learnable attention mechanism: <InlineMath math="\mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}."/>
            </p>
            <p> 
                After enabling nonlinearity through LeakyReLU, the attention coefficient is normalized across node features via softmax activation 
                (<InlineMath math="\theta = 0.2"/>):
            </p>
            <BlockMath math="\alpha_{ij} = \frac{\text{exp}(\text{LeakyReLU}(\beta_{ij}))}{\sum_{k \in \mathcal{N}_i} \text{exp}(\text{LeakyReLU}(\beta_{ij}))},"/>
            <p>where <InlineMath math="\mathcal{N}_i"/> represents the neighborhood of the node, determined by the adjacency matrix.</p>
            <p>
                Finally, the resulting node embeddings <InlineMath math="\mathbf{h'} = {\{\mathbf{h'}_1, \mathbf{h'}_2, \cdots, \mathbf{h'}_N}\}, \mathbf{h'}_i \in \mathbb{R}^{F'}"/> 
                are updated by aggregating across <InlineMath math="H"/> attention heads via average pooling:
            </p>
            <BlockMath math="\mathbf{h}_i' = \frac{1}{H}\sum_{k=1}^H\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j \in \mathbb{R}^{H \cdot F}."/>
            <p>
                This sequence is repeated for each time slice <InlineMath math="[t-\delta_t,\cdots,t]"/>, producing a temporal sequence 
                <InlineMath math="\mathbf{H}_{v,t-\delta_t:t} \in \mathbb{R}^{\delta_t\times d_h}"/>, for each node <InlineMath math="v"/>.
            </p> 
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Layer 2: Sequencing the Graphs</h3>
            <p>
                While the structural layer captures node representations that reflect their semantic relationships within the corporate 
                network at a given day <InlineMath math="t"/>, these relationships are constantly evolving. Stock features like price, 
                volatility, and news sentiment fluctuate daily, while institutional shareholders will enter or exit positions over time, 
                reshaping the network. Therefore, it is essential to extract the temporal evolution of node and edge features from the 
                graph snapshot sequence. 
                To accomplish this task, we employ the Transformer architecture with ALiBi to introduce an inductive bias toward recent 
                graph snapshots. The Transformer has become a dominant architectural choice in many domains––particularly that of 
                natural language processing and computer vision––due to its capacity for parallelization and contextual token embeddings. 
                Its effectiveness for learning sequential data is repurposed here to model the temporal evolution of graph snapshots. 
            </p>
            <img
                src={TransformerDiagram}
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
                The sequence of node embeddings over time <InlineMath math="\mathbf{H}_{v,t-\delta_t:t}"/> is passed through a 
                Transformer with ALiBi for temporal modeling. The temporal attention computes the learned projections of embeddings 
                into Query, Key, and Value spaces:
            </p>
            <BlockMath math="\mathbf{Q} = \mathbf{W}_\mathbf{Q}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}}, \nonumber \qquad
                            \mathbf{K} = \mathbf{W}_\mathbf{K}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}}, \nonumber \qquad
                            \mathbf{V} = \mathbf{W}_\mathbf{V}\mathbf{H}_{\mathbf{v},\mathbf{t}-\delta_{\mathbf{t}:\mathbf{t}}},"/>
            <p>
                where <InlineMath math="\mathbf{W_{\mathbf{Q}}}"/>, <InlineMath math="\mathbf{W_{\mathbf{K}}}"/>, and <InlineMath math="\mathbf{W_{\mathbf{V}}}"/> 
                are learned weight matrices. The temporal representation of the node is then obtained as follows: 
            </p>
            <BlockMath math="\mathbf{Z} = \text{softmax}\left(\frac{\mathbf{QK}^{\top}}{\sqrt{d_k}} + m \cdot \mathbf{P} + \mathbf{M}\right)\mathbf{V},"/>
            <p>
                where <InlineMath math="m\cdot \mathbf{P}"/> is the ALiBi term with <InlineMath math="m"/> controlling the strength of the bias, 
                and <InlineMath math="\mathbf{M}"/> acts as a causal mask, ensuring the model attends only to past and present time steps. 
            </p>
            </>
        }
        />
        <Section
        title=""
        content={
            <>
            <h3>Layer 3: Prediction</h3>
            <p>
                Each stock node is equipped with a dedicated linear layer that predicts its <strong>excess return</strong>: 
            </p>
            <BlockMath math="\hat{y}_{vt} = \mathbf{W}_1\mathbf{z}_{vt}+\mathbf{b}_1,"/>
            <p>
                Ground truth is defined to be the excess return relative to the S&P 500 index benchmark: 
            </p>
            <BlockMath math="y_{it} = \frac{p_{i,t+1}\;-\;p_{i,t}}{p_{i,t}} - \frac{SPX_{i,t+1}\;-\;SPX_{i,t}}{SPX_{i,t}}."/>
            </>
        }
        />
        <Section
        title="Experimental Design"
        content={
            <>
            <h3>Datasets</h3>
            <p>
                We leverage data from the Bloomberg Terminal and London Stock Exchange Data & Analytics to construct time-series 
                datasets encompassing market performance, company valuation metrics, financial news sentiment, ESG assessments, 
                ownership structures, and other categorical attributes. Industry node features are derived from the market 
                performance of the XLK Exchange-Traded Fund, which serves as a proxy for the Information Technology sector. 
                Prior to analysis, all features are normalized, winsorized, and subjected to dimensionality reduction via Principal 
                Component Analysis (PCA). 
            </p>
            <h3>Backtesting</h3>
            <p>
                We adopt a rolling window backtesting strategy to evaluate the model's performance across the varying market conditions 
                represented in the period from January 4th, 2019 to December 15th, 2022. The full historical dataset is divided into 
                overlapping windows that segment 6 months for training, 2 months for validation, and 2 months for testing. The Information 
                Coefficient (IC), Information Ratio (IR), Cumulative Return (CR), and Precision@K (K=30%) are computed for each cycle, and 
                the window advances by the length of the testing period. To prevent overfitting, each model instance is trained for 600 
                epochs with early stopping.
            </p>
            <h3>Metrics</h3>
            <p>
                We select the following standard financial metrics to assess the effectiveness of OmniGNN:
            </p>
            <ul>
                <li><strong>Information Coefficient (IC)</strong> measures the <em>Spearman rank correlation</em> between the model's 
                predicted and ground truth excess returns; </li>
                <li><strong>Information Ratio (IR)</strong> measures the <em>risk-adjusted performance</em> of the model, or the average 
                return of the top-K predicted stocks each day normalized by its volatility;</li>
                <li><strong>Cumulative Return (CR)</strong> is the <em>accumulated return</em> over the test period achieved by taking daily 
                long positions in the top-K% of stocks ranked by predicted excess return;</li>
                <li><strong>Precision@K</strong> measures the proportion of the top-K predicted stocks whose excess returns exceed the benchmark index.</li>
            </ul>
            <p>
                Model performance on the full historical dataset is evaluated by averaging the values of the model's performance metrics across all testing windows.
            </p>
            </>
        }
        />
        <Section
        title="Results"
        content={
            <>
            <h3>Baselines</h3>
            <p>
               The Transformer model is a time-series method that captures temporal dependencies in node features, but doesn't depend on graph structure. 
               The GAT model is insensitive to temporal dynamics, but it effectively captures local relational patterns in the knowledge graph. The performance
               of our model as compared to baseline models is presented in Table 1 and illustrated in Figure 2. The values in parentheses denote the relative 
               improvement of OmniGNN over each model.
            </p>
            <img
                src={BaselineResults}
                alt="Baseline Results"
                style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }}
            />
            <img
                src={BaselineResultsGraph}
                alt="Baseline Results Graph"
                style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }}
            />
    
            <p>
                The experiments show that the Transformer model outperforms the GAT in all metrics, suggesting that the time-series data stored by the 10-stock 
                corporate network is more informative than its relational patterns for the task of excess return prediction. Furthermore, OmniGNN achieves superior 
                predictive performance across all metrics, likely because it combines temporal and structural reasoning to learn how the complex relationships 
                between stocks and industries evolve over time.
            </p>
            <p>
                OmniGNN's reported IR is 0.0767, which means that the model generates little positive risk-adjusted return. Still, the daily IC of 0.0673 indicates 
                a moderate, meaningful predictive signal. Using the top-K portfolio strategy, OmniGNN's average CR of 2.18% over 2-month testing periods annualizes 
                to roughly 13.8%. For comparison, the average annualized CR of the SPX index over the same testing periods is 10.04%. Overall, our model demonstrates 
                a modest but consistent ability to generate daily alpha across varying market regimes, including periods of heightened volatility like the COVID-19 
                pandemic. Furthermore, the strength of our model's performance is underscored by the fact that it predicts a basket of 10 tech stocks with an average 
                daily volatility of 0.0228, which is 1.58 times higher than the SPX index's average daily volatility over the same period. 
            </p>
            <h3>Case Study: COVID-19</h3>
            <p>
               To test the impact of the global node on model performance, we perform an ablation experiment by removing the industry 
               node, the <InlineMath math="\mathcal{E}_{\mathcal{SI}}"/> edge, and the corresponding <InlineMath math="\mathcal{SIS}"/> metapath. Notably, the testing 
               window covers the onset of the COVID-19 pandemic (March 4 – May 4, 2020), allowing us to evaluate model performance during a time of extreme market 
               volatility.
            </p>
            <img
                src={Ablation}
                alt="Case Study Results"
                style={{ width: '100%', borderRadius: '12px', marginTop: '1rem' }}
            />
            <p>
                The inclusion of the industry node and its corresponding edges and metapaths leads to improvements in all metrics during the onset of the COVID-19 
                pandemic. This is likely because during the quarantine and ensuing financial crisis that characterized the pandemic, macroeconomic shocks impacted companies
                 through their industry-specific risk exposures, which are represented by the <InlineMath math="\mathcal{SIS}"/> metapath. Interestingly, the case study 
                 reveals that while OmniGNN's average predictive power degraded during the onset of COVID-19 (evidenced by a negative IC), it was able to select the top-K 
                 stocks more accurately than it did from 2019-2022, resulting in greater IR and CR metrics. 
            </p>
            </>
        }
        />
        <Section
        title="Future Work"
        content={
            <>
            <p>
                Building on the insights from our current model, there remain several promising directions we aim to explore in future work. These include architectural 
                modifications, alternative supervision signals, and improved graph construction strategies that may further enhance predictive performance and interpretability.
            </p>
            <ul>
                <li><strong>Time-Lag Incorporation.</strong> In predicting future stock price behavior, OmniGNN relies on technical analysis; however, this approach doesn't account 
                for the time lag inherent to financial markets. For example, investor sentiment expressed in the news on a given day will not necessarily affect stock prices 
                immediately. Matching the lag rate of the indicators, therefore, could improve the predictive accuracy of OmniGNN</li>
                <li><strong>Edge Attributes.</strong> Sourcing broader relationship types (industry, subsidiary, people, and product relations) through open-source knowledge graphs 
                can be used to increase the dimensions of edge attributes. This approach, as explored by IBM Research Japan, enhances the expressiveness of the graph structure. 
                Subsequently, dimensionality reduction techniques such as PCA can be applied to distill more meaningful and compact representations of inter-entity relationships.</li>
                <li><strong>Advanced Sentiment Analysis.</strong> OmniGNN utilizes a text processing library <strong>TextBlob</strong> to obtain polarity scores for each news article. 
                However, the default models in <strong>TextBlob</strong> lack accuracy for specialized domains, calling for more advanced language processing techniques.</li>
                <li><strong>New Node Types.</strong> The next steps also include designing new node types, such as a macroeconomic node that stores several macroeconomic indicators 
                and determines explicit market regimes.</li>
            </ul>
            </>
        }
        />
        <Section
        title="Conclusion"
        content={
            <>
            <p>
                Our proposed model, OmniGNN, makes several contributions to the GNN literature. With multi-dimensional metapaths defining longer-hop relationships through the heterogeneous 
                network schema, OmniGNN is able to apply attention weights to learn structurally meaningful representations of stock nodes while filtering out noise from less informative 
                connections. As shown through ablation experiments, these new paths improves predictive performance of downstream tasks such as node-level regression. Furthermore, these 
                metapaths and node features evolve across time. OmniGNN repurposes the innovative Transformer mechanism to sequence graph snapshots, which transforms the hidden embeddings 
                into ones that account for longer-range temporal dependencies between trading days. Finally, OmniGNN's structural innovation, the global node intermediary, densifies graphs 
                with a star topology overlay, enabling efficient message propagation during periods of global disruption. Our results show that OMNI-GNN achieves consistent predictive power 
                for the daily excess returns for 10 relatively volatile tech stocks during a 3-year period with greatly varied market regimes. Future work will focus on extending our 
                experiments to incorporate entire corporate networks, such as the entire Information Technology sector or S&P 500.
            </p>
            </>
        }
        />
        <Section
        title="Acknowledgments"
        content={
            <>
            <p>
                This research was conducted as part of the Columbia Summer Undergraduate Research Experiences in Mathematical Modeling program, hosted and supported by the Columbia University 
                Department of Mathematics. We are grateful to Professor George Dragomir, Professor Dobrin Marchev, and Vihan Pandey for their support throughout the project. 
            </p>
            </>
        }
        />
        <Footer />
    </div>
  );
}

export default App;