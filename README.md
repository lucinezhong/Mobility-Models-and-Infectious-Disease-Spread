<h1> Mobility Models and Infectious Disease Spread on Mobility Networks </h1>

<p>
This repository includes codes and analyses of the collective mobility model, individual mobility model, infectious disease model on the human mobility network, and optimization algorithm on controlling network infections.

<h1> Collective Mobility Model </h1>

<h2>Gravity Model</h2>

Inspired by Newton's law of gravitation, George K. Zipf proposed an equation to model mobility flows. The model assumes that the number of trips originating from location i is proportional to its population, the attractiveness of destination j is proportional to its population at the cost of distance, 
<p align="center"> <I> T<sub>ij</sub> = K M<sub>i</sub>N<sub>j</sub> f(r<sub>ij</sub>)

<p>where K is constant, the M<sub>i</sub>  and N<sub>j</sub> repsectively resprsents the massesses. f(r<sub>ij</sub>) represetens the descreasing fucton of idstance. The most common used form for masses as M<sub>i</sub>=P<sub>i</sub><sup>&alpha;</sup> and  N<sub>j</sub>=P<sub>j</sub><sup>&alpha;</sup>.  </p>

<h2>Radiation Model</h2>
The Radiation Model, proposed by Filippo Simini et al, presents a parameter approach to estimating commuting flues between two locations. Unlike traditional gravity-based models, which rely on tunable parameters to fit empirical data, the Radiation Model derives mobility flows from population distribution alone, making it a more universal and scalable method. The model assumes that the number of trips from origin location i to destination j depends on not only the population of two locations but also the presence of alternative opportunities in the surrounding areas. By introducing S<sub>ij</sub>, the population within the radius of  r<sub>ij</sub> centered around location i, the model predicts the flow as
<p align="center"> <i>E(T<sub>ij</sub>) = T<sub>i</sub>M<sub>i</sub>N<sub>j</sub> / (M<sub>i</sub> + S<sub>ij</sub>)(M<sub>i</sub> + N<sub>j</sub> + S<sub>ij</sub>)</i> </p>


<h2>Visitation Model</h2>

Dr. Schläpfer and Dr. Dong, through extensive data analysis, identified a key relationship governing the frequency and spatial distribution of human visits. Their research reveals that the number of visitors  N<sub>i</sub> (r,f) at a location systematically decreases with travel distance r and travel frequency f. The visitation density is defined as 
&rho;<sub>i</sub>,
<p align="center"> <i> &rho;<sub>i</sub> (r,f)= N<sub>i</sub> (r,f)/A(r)= &mu;<sub>i</sub>/(rf)<sup>&eta;</sup> </i>  </p>

The average number of trips of those individuals live in location i to destination j cand be estimated as 
<p align="center"> <i> T<sub>ij</sub> &approx; &mu;<sub>j</sub>A<sub>i</sub>/r<sub>ij</sub><sup>2</sup>ln(f<sub>max</sub>/f<sub>min</sub>) </i>  </p>
where A<sub>i</sub> is the area of original location, r<sub>ij</sub> is the distance. &mu; is the location-specific attractiveness.
  
<h2>**Comparison</h2>
We use the Sørensen similarity index (SSI) to measure the similarities between estimated flows and true flows between two locations. The SSI is between 0 and 1, higher value indicates higher similarity/accuracy.
<p align="center"> <i> SSI=2 &sum; <sub>ij</sub> min (T<sub>ij</sub> <sup>model</sup>, T<sub>ij</sub><sup>data</sup>)/(&sum; T<sub>ij</sub> <sup>model</sup>+ &sum; T<sub>ij</sub><sup>data</sup>)

*** Some input datasets are fabricated.
  
<h1> Individual Mobility Model </h1>

<h2>EPR Model</h2>

<p> The EPR (Exploration and Preferential Return) model (Proposed by Chaoming Song et al) is a classical individual mobility model that describes human mobility dynamics based on two fundamental behavioral tendencies. This model has successfully captured individual mobility scaling, including 1) unique locations S(t)&sim;t<sup>&mu;</sup> ; 2) Zipf's law of visitation frequency; 3) ultraslow diffusion </p>
<p> Exploration – With probability P=&rho;S<sup>-&gamma;</sup>, the individual will explore a new location.</p> 
<p> Preferential Return – With probability 1-P, the individual will return to a previously visited location i according to the locations' past visitation frequency f<sub>i<sub> .</p> 

Many variations of this model have been proposed, such as the d-EPR and p-EPR models.

<h2>d-EPR Model</h2>
Unlike the EPR model, in the exploration phase individuals randomly select a new location, the d-EPR model proposes that individuals visit new locations based on the gravity model, w with probability P<sub>ij</sub>.


<h2>PEPR Model</h2>
Unlike the EPR model, in the exploration phase individuals randomly select a new location, the Preferential Exploration and Preferential Return (PEPR) model that when individuals explore new locations, they tend to favor areas that are frequently visited. Specifically, exploration direction is biased toward regions with high visitation with distribution P(&theta; ; R,v).


<h2>Switch Model</h2>

<h2>**Comparison</h2>

<h1> Meta-population Model for Simulating Infectious Disease Spread </h1>

<h2>  SIR-Metapopulation Model </h2>

<h2>  SIR-Metapopulation Model with Multiple OLs </h2>


<h1> Optimization Algorithms on Controlling Infectious Disease Spread </h1>

<h2> genetic algorithm </h2>h2> 
<p>References: </p>
<p>[1] Barbosa, H., Barthelemy, M., Ghoshal, G., James, C. R., Lenormand, M., Louail, T., ... & Tomasini, M. (2018). Human mobility: Models and applications. Physics Reports, 734, 1-74.</p>
<p>[2] Belik, V., Geisel, T., & Brockmann, D. (2011). Natural human mobility patterns and spatial spread of infectious diseases. Physical Review X, 1(1), 011001.</p>
<p>[3] Simini, F., González, M. C., Maritan, A., & Barabási, A. L. (2012). A universal model for mobility and migration patterns. Nature, 484(7392), 96-100.</p>
<p>[4] Schläpfer, M., Dong, L., O’Keeffe, K., Santi, P., Szell, M., Salat, H., ... & West, G. B. (2021). The universal visitation law of human mobility. Nature, 593(7860), 522-527.</p>
<p>[5] Song, C., Koren, T., Wang, P., & Barabási, A. L. (2010). Modelling the scaling properties of human mobility. Nature physics, 6(10), 818-823.</p>
<p> [6] Pappalardo, L., Simini, F., Rinzivillo, S., Pedreschi, D., Giannotti, F., & Barabási, A. L. (2015). Returners and explorers dichotomy in human mobility. Nature communications, 6(1), 8166. </p>
<p> [7] Brockmann, D., & Helbing, D. (2013). The hidden geometry of complex, network-driven contagion phenomena. science, 342(6164), 1337-1342. </p>
<p> [8] Zhong, L., Diagne, M., Wang, W., & Gao, J. (2021). Country distancing increase reveals the effectiveness of travel restrictions in stopping COVID-19 transmission. Communications Physics, 4(1), 121. </p>
<p> [9] Wang, Y., Zhong, L., Du, J., Gao, J., & Wang, Q. (2022). Identifying the shifting sources to predict the dynamics of COVID-19 in the US. Chaos: An Interdisciplinary Journal of Nonlinear Science, 32(3).
<p> [10] Zhong, L., Dong, L., Wang, Q., Song, C., & Gao, J. (2024). Universal spatial inflation of human mobility. arXiv preprint arXiv:2406.06889. <p>




