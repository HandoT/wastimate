"""
Wastimate is a specialized Python-based framework designed for simulating, 
managing, and analyzing the behavior of radioactive waste over time. It focuses
on simulating complex processes such as isotopic decay, waste sorting,
conditioning, and transportation within a dynamic system of interconnected
nodes. This tool can be used to assess long-term waste management strategies,
ensure regulatory compliance, and optimize the handling of radioactive waste
in facilities like nuclear power plants or waste repositories.
The code leverages object-oriented design principles, with core classes such
as Node, Package, Order, and Universe to model the flow of radioactive
materials and simulate the impact of various waste handling operations over
time.


NB! Wastimate requires decay chain data to run, place the OpenMC's decay chain
data in the same folder as the .py. (Data that is formatted to suit Wastimate
can be found here https://www.dropbox.com/scl/fi/zexn5i1fuke8u50thj7c2/decay_chains_endfb71.xml?rlkey=xsukcszkrpagt6skcv8ara47o&st=sbx4yi28&dl=0)
"""