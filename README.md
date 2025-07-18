<h1> Mobility Models and Infectious Disease Spread on Mobility Networks </h1>

<p>
This repository includes codes and analyses of the collective mobility model, individual mobility model, infectious disease model on the human mobility network, and  algorithms on controlling network infections.
 
<p align="center">
	<img src="all_diagram.png" width="700" height="350" />
<p>

<h1> Collective Mobility Model </h1>

<h2>Gravity Model</h2>

Inspired by Newton's law of gravitation, George K. Zipf proposed an equation to model mobility flows. The model assumes that the number of trips originating from location i is proportional to its population, the attractiveness of destination j is proportional to its population at the cost of distance, 
<p align="center"> <I> T<sub>ij</sub> = K M<sub>i</sub>N<sub>j</sub> f(r<sub>ij</sub>)

<p>where K is constant, the M<sub>i</sub>  and N<sub>j</sub> repsectively resprsents the massesses. f(r<sub>ij</sub>) represetens the descreasing fucton of idstance. The most common used form for masses as M<sub>i</sub>=P<sub>i</sub><sup>&alpha;</sup> and  N<sub>j</sub>=P<sub>j</sub><sup>&alpha;</sup>.  </p>

<h2>Radiation Model</h2>
The Radiation Model（Filippo Simini et al., 2012), presents a parameter-free approach to estimating commuting flues between two locations. Unlike traditional gravity-based models, which rely on tunable parameters to fit empirical data, the Radiation Model derives mobility flows from population distribution alone, making it a more universal and scalable method. The model assumes that the number of trips from origin location i to destination j depends on not only the population of two locations but also the presence of alternative opportunities in the surrounding areas. By introducing S<sub>ij</sub>, the population within the radius of  r<sub>ij</sub> centered around location i, the model predicts the flow as
<p align="center"> <i>E(T<sub>ij</sub>) = T<sub>i</sub>M<sub>i</sub>N<sub>j</sub> / (M<sub>i</sub> + S<sub>ij</sub>)(M<sub>i</sub> + N<sub>j</sub> + S<sub>ij</sub>)</i> </p>


<h2>Visitation Model</h2>

Dr. Markus and Dr. Dong (Schläpfer Markus et al., 2021), through extensive data analysis, identified a key relationship governing the frequency and spatial distribution of human visits. Their research reveals that the number of visitors  N<sub>i</sub> (r,f) at a location systematically decreases with travel distance r and travel frequency f. The visitation density is defined as 
&rho;<sub>i</sub>,
<p align="center"> <i> &rho;<sub>i</sub> (r,f)= N<sub>i</sub> (r,f)/A(r)= &mu;<sub>i</sub>/(rf)<sup>&eta;</sup> </i>  </p>

The average number of trips of those individuals live in location i to destination j cand be estimated as 
<p align="center"> <i> T<sub>ij</sub> &approx; &mu;<sub>j</sub>A<sub>i</sub>/r<sub>ij</sub><sup>2</sup>ln(f<sub>max</sub>/f<sub>min</sub>) </i>  </p>
where A<sub>i</sub> is the area of original location, r<sub>ij</sub> is the distance. &mu; is the location-specific attractiveness.
  
<h2>**Comparison</h2>
We use the Sørensen similarity index (SSI) to measure the similarities between estimated flows and true flows between two locations. The SSI is between 0 and 1, higher value indicates higher similarity/accuracy.
<p align="center"> <i> SSI=2 &sum; <sub>ij</sub> min (T<sub>ij</sub> <sup>model</sup>, T<sub>ij</sub><sup>data</sup>)/(&sum; T<sub>ij</sub> <sup>model</sup>+ &sum; T<sub>ij</sub><sup>data</sup>)

Results: The Gravity model reports an SSI of 0.56, while the Radiation model records the lowest SSI at 0.46. Conversely, the Visitation model achieves the highest SSI of 0.7.


<p align="center">
	<img src="comparison_figure.png" width="900" height="350" />
<p>
  
<h1> Individual Mobility Model </h1>

<h2>EPR Model</h2>

<p> The EPR (Exploration and Preferential Return) model (Song Chaoming et al., 2010) is a classical individual mobility model that describes human mobility dynamics based on two fundamental behavioral tendencies. This model has successfully captured individual mobility scaling, including 1) unique locations S(t) ~ t<sup>&mu;</sup> ; 2) Zipf's law of visitation frequency; 3) ultraslow diffusion </p>
<p> Exploration – With probability P=&rho;S<sup>-&gamma;</sup>, the individual will explore a new location.</p> 
<p> Preferential Return – With probability 1-P, the individual will return to a previously visited location i according to the locations' past visitation frequency f<sub>i<sub> .</p> 

Many variations of this model have been proposed, such as the d-EPR and p-EPR models.

<h2>d-EPR Model</h2>
Unlike the EPR model, in the exploration phase individuals randomly select a new location, the d-EPR model (Luca Pappalardo et al. 2015) proposes that individuals visit new locations based on the gravity model, with probability P<sub>ij</sub>.


<h2>PEPR Model</h2>
Unlike the EPR model, in the exploration phase individuals randomly select a new location, the Preferential Exploration and Preferential Return (PEPR) model (Schläpfer Markus et al., 2021) states that when individuals explore new locations, they tend to favor areas that are frequently visited. Specifically, exploration direction is biased toward regions with high visitation with distribution P(&theta;; R,v).

<h2>Switch Model</h2>
This model is inspired by the observation that human mobility exhibits distinct spatial and topological characteristics: it is modular, and within each module, there are hub locations that facilitate revisitation. This configuration demonstrates high modularity but low clustering coefficients. To reproduce it, we (Lu Zhong et al., 2024; Lu Zhong et al., 2025) propose the 'switch model' to enable mechanisms that govern transitions both within and across modules.

<h2>**Comparison</h2>
The EPR model generates trajectories with shorter distances traveled around the home, while the d_EPR model generates trajectories that follow the probability of country outflows, resulting in longer-distance paths and greater connectivity. The switch model enables modular-like behavior in human trajectory. 

<p align="center">
	<img src="comparison_individual.png" width="900" height="300" />
<p>

<h1> Meta-population Model for Simulating Infectious Disease Spread </h1>

<h2>  SIR-Metapopulation Model </h2>
The Susceptible-Infectious-Recovered (SIR) metapopulation model is a mathematical framework used to study the spread of infectious diseases across multiple, interconnected populations. Unlike the classic SIR model, which assumes a single well-mixed population, the metapopulation approach accounts for spatial heterogeneity by dividing the population into distinct subpopulations (or patches) connected by mobility or migration dynamics. In this framework let there be n regions (subpopulations), each governed by the standard SIR dynamics, where individuals exist in three states: susceptible s<sub>n</sub>, infected i<sub>n</sub>,  removed r<sub>n</sub>. Disease transmission within each subpopulation follows local interactions, while the inter-subpopulation spread is driven by travel dynamics, represented by the movement flow matrix P<sub>mn</sub>,  which quantifies the mobility of individuals between regions m and n,

<p align="center"> <i> ṡ<sub>n</sub>=-&alpha; s<sub>n</sub>i<sub>n</sub> &sigma;(i<sub>n</sub>/&varepsilon;) +&gamma; &sum;<sub>m &ne; n</sub> P<sub>mn</sub> (s<sub>m</sub>-s<sub>n</sub>)
<p align="center"> <i> i̇<sub>n</sub>=&alpha; s<sub>n</sub>i<sub>n</sub> &sigma;(i<sub>n</sub>/&varepsilon;) -&beta;i<sub>n</sub> +&gamma; &sum;<sub>m &ne; n</sub> P<sub>mn</sub> (i<sub>m</sub>-i<sub>n</sub>)

witch single initial outbreak location k, given as:

s<sub>k</sub>=s<sub>k</sub><sup>real</sup>, i<sub>k</sub>=i<sub>k</sub><sup>real</sup>, r=r<sub>k</sub><sup>real</sup>.

<b>Effective Distance</b> (Brockmann D., Helbing D. 2013): By defining the effective distance d<sub>mn</sub>=1-logP<sub>mn</sub>, the distance from an arbitrary node n to node m is the lengths of shortest path ( &Gamma;) of effective distance,
<p align="center"> D<sub>mn</sub>=&sum;<sub>(i,j)&in;&Gamma;</sub>   d<sub>ij</sub>

From the initial outbreak location k, effective distance predicts arrival times,
<p align="center"> D<sub>mk</sub> &sim; T<sub>m</sub><sup>arrival</sup>



<h2>  Metapopulation Model with Multiple OLs </h2>
As intervention measures and travel restrictions vary across regions and outbreak locations shift with differing infection levels, we propose the SIR-Metapopulation Model with Multiple Outbreak Locations (OLs) and a dynamic flow matrix P<sub>mn</sub> (t). This model accounts for evolving mobility patterns and region-specific control strategies, enabling a more adaptive representation of disease spread in changing environments.

<p align="center"> <i> ṡ<sub>n</sub>=-&alpha;(t) s<sub>n</sub>i<sub>n</sub> &sigma;(i<sub>n</sub>/&varepsilon;) +&gamma;(t) &sum;<sub>m &ne; n</sub> P<sub>mn</sub>(t) (s<sub>m</sub>-s<sub>n</sub>)
<p align="center"> <i> i̇<sub>n</sub>=&alpha;(t) s<sub>n</sub>i<sub>n</sub> &sigma;(i<sub>n</sub>/&varepsilon;) -&beta;(t)i<sub>n</sub> +&gamma;(t) &sum;<sub>m &ne; n</sub> P<sub>mn</sub>(t) (i<sub>m</sub>-i<sub>n</sub>)

witch changing outbreak location set N<sub>I</sub>(t), given as:

s<sub>k</sub>=s<sub>k</sub><sup>real</sup>, i<sub>k</sub>=i<sub>k</sub><sup>real</sup>, r=r<sub>k</sub><sup>real</sup>, &forall; k &in; N<sub>I</sub>(t)

<b>Multiple-OLs Effective Distance</b> (Lu Zhong et al. 2021): By defining the effective distance d<sub>mn</sub>=1-logP<sub>mn</sub>, the distance from mutilpe outbreak locations N<sub>I</sub> to node m is computed as,
<p align="center"> D<sub>m|N<sub>I</sub> </sub>=log (1/ &sum;<sub>n<sub>i</sub> &in;  N<sub>I</sub> </sub> 1/e<sup>d<sub>m|n<sub>i</sub> </sup> )

From the outbreak location set N<sub>I</sub>, effective distance predicts arrival times,
<p align="center"> D<sub>m|N<sub>I</sub>  &sim; T<sub>m</sub><sup>arrival</sup>

<p align="center">
	<img src="Infection_vesus_distance.png" width="700" height="350" />
<p>
	
<h2>  POI-Metapopulation Model </h2>
Chang et al. (2021) proposed a bipartite graph model that links Census Block Groups (CBGs), where people reside, to Points of Interest (POIs) they visit. In this model, the researchers overlaid the metapopulation to the bipartite to include disease transmission within the CBGs and also  disease transmission by visiting the POIs. 

<h2>  Agent-based model </h2>
Agent-based epidemic models (ABMs) are powerful computational tools used to simulate the spread of infectious diseases within a population by modeling the behaviors and interactions of individual entities, or "agents." Unlike traditional compartmental models that rely on population-level assumptions and averaged dynamics, ABMs capture heterogeneity in individual attributes, such as age, behavior, location, and health status, as well as complex contact structures and stochastic transmission events.

<p align="center">
	<img src="agent_model_diagram.png" width="700" height="300" />
<p>
In real-world populations, individuals don’t mix randomly—people interact within structured social, spatial, and organizational patterns. Networks represent these patterns by modeling agents as nodes and their interactions or relationships (e.g., physical contact, shared spaces, communication) as edges. By embedding agents in a network, ABMs can more realistically simulate how diseases spread through actual contact pathways
	
<p align="center">
	<img src="agent_model_network.png" width="700" height="300" />
<p>

<p> References: </p>
<p>[1] Barbosa, H., Barthelemy, M., Ghoshal, G., James, C. R., Lenormand, M., Louail, T., ... & Tomasini, M. (2018). Human mobility: Models and applications. Physics Reports, 734, 1-74.</p>
<p>[2] Belik, V., Geisel, T., & Brockmann, D. (2011). Natural human mobility patterns and spatial spread of infectious diseases. Physical Review X, 1(1), 011001.</p>
<p>[3] Simini, F., González, M. C., Maritan, A., & Barabási, A. L. (2012). A universal model for mobility and migration patterns. Nature, 484(7392), 96-100.</p>
<p>[4] Schläpfer, M., Dong, L., O’Keeffe, K., Santi, P., Szell, M., Salat, H., ... & West, G. B. (2021). The universal visitation law of human mobility. Nature, 593(7860), 522-527.</p>
<p>[5] Song, C., Koren, T., Wang, P., & Barabási, A. L. (2010). Modelling the scaling properties of human mobility. Nature physics, 6(10), 818-823.</p>
<p> [6] Pappalardo, L., Simini, F., Rinzivillo, S., Pedreschi, D., Giannotti, F., & Barabási, A. L. (2015). Returners and explorers dichotomy in human mobility. Nature communications, 6(1), 8166. </p>
<p> [7] Brockmann, D., & Helbing, D. (2013). The hidden geometry of complex, network-driven contagion phenomena. science, 342(6164), 1337-1342. </p>
<p> [8] Zhong, L., Diagne, M., Wang, W., & Gao, J. (2021). Country distancing increase reveals the effectiveness of travel restrictions in stopping COVID-19 transmission. Communications Physics, 4(1), 121. </p>
<p> [9] Wang, Y., Zhong, L., Du, J., Gao, J., & Wang, Q. (2022). Identifying the shifting sources to predict the dynamics of COVID-19 in the US. Chaos: An Interdisciplinary Journal of Nonlinear Science, 32(3).</p>
<p> [10] Zhong, L., Dong, L., Wang, Q.R., Song, C. and Gao, J., 2025. Universal expansion of human mobility across urban scales. Nature Cities, pp.1-5.
<p> [11] Zhong, L., Dong, L., Wang, Q., Song, C., & Gao, J. (2025). Switching exploration modes in human mobility. arXiv preprint </p>
<p> [12] Chang, S., Pierson, E., Koh, P. W., Gerardin, J., Redbird, B., Grusky, D., & Leskovec, J. (2021). Mobility network models of COVID-19 explain inequities and inform reopening. Nature, 589(7840), 82-87.</p>




