# Source audit

Pinned arXiv v1 source and PDF; clean-room local execution only.

## Claim 1 source locations

```text
main.tex: \icmltitlerunning{Graph Neural Networks Are Not Continuous Across Graph Resolutions}
main.tex:   \icmltitle{Graph Neural Networks Are Not Continuous Across Graph Resolutions}
main.tex: We show that contrary to conventional wisdom in the community, graph neural networks (GNNs) are not continuous with respect to all natural modes of graph convergence. As a result, GNNs may generate substantially different latent repre-\\ sentations for graphs that are very similar. In par-\\ ticular
main.tex: formation-propagation schemes. Building on this insight, we then derive a principled modification to standard GNN architectures which equips models with continuity across scales. The proposed modification enables consistent integration of distinct resolutions and reliable generalization between them
main.tex:  In particular we demonstrate that standard GNN architectures can produce \emph{drastically different latent representations} for graphs that describe the same underlying object at different \emph{resolution scales}. 
main.tex: As a consequence, models trained on graphs at one resolution, for
main.tex: instance, \emph{fail} to generalize to equivalent graphs at another resolution.
main.tex:  This observation reveals a fundamental limitation of existing GNN architectures. It also motivates the development of modifications to existing architecures, in order to render them \emph{continuous} across graph resolution scales.  
main.tex:     underappreciated failure mode of standard GNNs, i.e., \textbf{a lack of continuity across graph resolutions}.
main.tex: From a physical perspective, describing a molecule at the level of interacting atoms corresponds to a specific choice of resolution scale:
main.tex: Interactions of individual protons and neutrons inside the various atomic nuclei are discarded.\footnote{ At an even higher resolution scale, also interactions of quarks and gluons (in turn making up protons and neutrons) are neglegted.} 
main.tex: we additionally also consider a version of QM$7$ where we lower the resolution scale even further:
main.tex: QM$7_{\text{coarse}}$ dataset models  data obtained from a resolution-limited observation process
main.tex: Using the high-resolution graphs $\{G\}$ of QM$7$ and the low-resolution graphs $\{\underline{G}\}$ in  QM$7_{\text{coarse}}$, we then investigate the
main.tex: by confronting models during inference with a resolution-scale different from the one  they were trained on. 
main.tex: Mean-absolute-errors (MAEs) during inference increase significantly, when going from a same-resolution  setting to a cross-resolution setting. 
main.tex: Corresponding cross-resolution MAEs 
main.tex: 	\caption{QM7 regression. Mean Absolute Error (MAE $\downarrow$) in kcal/mol for training and inference at different resolutions scales.}
main.tex: 		\textbf{Resolution} & \multicolumn{4}{c}{\textbf{MAE ($\downarrow$) on QM7 [kcal/mol]}} \\
main.tex: Embedding difference $\|F - \underline{F}\|$ across resolution scales averaged over 5 runs (mean$\pm$std). Lower is better ($\downarrow$).
main.tex: of graphs describing the same object on varying resolutions are significantly different. This in turn explains the inability to generalize across scales.
main.tex: \noindent It should be noted that in practice, this failure \emph{cannot} be overcome by augmenting the training set, as we have no way of generating faithful high-resolution descriptions given only lower resolution training data.
main.tex: 			The corresponding \textbf{Laplace-transform propagation matrix} is the matrix  $\psi(L) \in \mathbb{R}^{N \times N}$  arising as the Laplace transform (cf. Appendix \ref{app:ltf_matrices} for details) of  $\hat{\psi}$:
main.tex: 						Using Laplace-transform propagation matrices as in (\ref{eq:LT_integral}) together with the propagation rule (\ref{eq:global_laplacian_proagation}) in each layer leads to scale continuos networks: If $G_\omega \rightarrow\underline{G}$ in the sense of heat kernels, then $F_\omega \rightarrow \
main.tex: At first glance, propagation along Laplace-transform matrices as in 
main.tex: 		Let $F$ and $\underline{F}$ be the latent embeddings generated for a graph $G$ its coarsified version $\underline{G}$ by a (spectral or message passing) network employing Laplace transform propagation, as outlined in Section~\ref{subsec:sc_gnns}. With $\{\Psi_i(L) = \int_0^\infty \hat{\psi}_i(t)e^
main.tex: Laplace-transform propagation schemes are continuous as maps from the
main.tex: In Section \ref{sec:non-cont} we had identified lack of continuity as the obstruction to generalizing across scales. As verified above, graph neural networks based on Laplace-transform propagation \textit{are} continuous. Hence we expect them to map  similar graphs to similar latent embeddings. 
main.tex: 				Embedding difference $\|F - \underline{F}\|$ across resolution scales averaged over 5 runs (mean$\pm$std). Lower is better ($\downarrow$).
main.tex: we see that in cross-resolution settings the difference $\|F - \underline{F}\|$ of latent embeddings generated by  
main.tex: translate to small-to-negligible variations in prediction performance: As we infer from Table \ref{tab:qm7_resolvent_results} below, MAEs generated by Laplace-Transform based GNNs in the cross resolution setting are essentially the same as those corresponding to same-resolution settings.
main.tex: 	\caption{QM7 regression. Mean Absolute Error (MAE $\downarrow$) in kcal/mol for training and inference at different resolutions scales.}
main.tex: 	\textbf{Resolution} & \multicolumn{4}{c}{\textbf{MAE ($\downarrow$) on QM7 [kcal/mol] }} \\
main.tex: 						Both typical models and the Laplace-transform propagation based methods introduced in Section \ref{subsec:sc_gnns} were then trained on the same ($k$-fold expanded) train-set and asked to classify nodes in the ($k$-fold expanded) test-partition. 
main.tex: The classification accuracies of methods not employing Laplace-transform propagation decrease significantly with increasing clique size (cf.  Fig. \ref{fig:node_blowup}). We can understand the underlying reason for this using GCN as an Example (cf. Appendix \ref{app:limitprop} for discussions on oth
main.tex: This is not the case for Laplace-transform propagation based networks (using either resolvent or exponential propagation matrices): As Corrolary \ref{cor:main} elucidates, similarity of graphs for such models is not determined through the renormalized adjacency as for GCN. Rather it is governed by t
main.tex: high connectivity areas and therefore are able to retain a high classification accuracy as the resolution scale is increased.
main.tex: Finally, we consider the setting where graphs discretize a continuous manifold. In this regime, we require that as we increase the resolution scale, model outputs converge to those that would be produced by a model operating directly on the underlying continuous manifold.
main.tex:  the correlation between two nodes representing (the same two) points on the manifold to stabilize, as the resolution is increased. 
main.tex: 		(two resolutions).} 
main.tex:  As evident from Fig. \ref{fig:impulse_response} (a), this requirement is not fulfilled for standard GNNs, as  as neithers models' response stabilizes: For most GNNs the impuls response drops to zero, while for Lanczos it fluctuates as the mesh resolution (i.e. number of nodes) is increased. 
main.tex: 		\caption{Laplace-Transform GNN\\ impulse response}
main.tex: In contrast to that, the impulse response for the Laplace-transform based methods of Section \ref{subsec:sc_gnns} stays consistent as the mesh resolution is varied.
main.tex: showed that these modified models based on Laplace-Transform propagation can
main.tex: \section{Stability of latent representations generated by Laplace-transform based GNNs for graphs on the same node set }\label{app:gogginstheoremII_proof}
main.tex: \subsubsection{Spectral Laplace-transform methods}
main.tex: \section{Stability of latent representations generated by Laplace-transform based GNNs for graphs on differing resolution scales}\label{app:coarsification_prooc}
main.tex: In this section, we establish bounds on differences in latent representations generated for Laplace-transform based GNNs when confronted with graphs describing the same underlying object at multiple resolutions. We briefly recall the setting:
main.tex: We have a high resolution graph $G$ with associated Laplacian $L$ and node-feature matrix $X$. We also have a lower resolution graph $\underline{G}$, with associated Laplacian $\underline{L}$ and node-feature matrix $\underline{X}  := J^\downarrow X$ arising from the original node feature matrix $X$
main.tex: We are then initially interested in the following question: Suppose we have a node feature matrix $X$ on the graph $G$. We can generate node-level latent embeddings $\Xi(X)$ by feeding this node-wise information into a (Laplace-transform based) node-level GNN $\Xi$. How different is this outcome fro
main.tex: \subsubsection{Spectral Laplace-transform methods}\label{app:spectral_coarse_fine_results}
main.tex: Finally we recall from Section \ref{subsec:sc_gnns}, that the initial input $X^0$ into the Laplace-transform message passing layers arises  from the input features $X$ via an initial Laplace-transform propagation step as 
main.tex: 		Let $F$ and $\underline{F}$ be the latent embeddings generated for a graph $G$ its coarsified version $\underline{G}$ by a (spectral or message passing) network employing Laplace transform propagation, as outlined in Section~\ref{subsec:sc_gnns}. With $\{\Psi_i(L) = \int_0^\infty \hat{\psi}_i(t)e^
main.tex: 	Using this as an upper bound in Theorem \ref{subsec:sc_gnns} shows that embeddings $F, \underline{F}$ of graphs describing the same molecule at different resolution scales are similar. This explains the ability to generalize between scales.
main.tex: Equation (\ref{eq:cauchy}) then guarantees that the corresponding graph level latent embeddings form a Cauchy sequence. Since the latent space $\mathbb{R}^d$ is complete, we thus have convergence of the latent embeddings $F_N$ (indexed by the mesh size $N$) towards a limit embedding ($F_N \rightarro
```
