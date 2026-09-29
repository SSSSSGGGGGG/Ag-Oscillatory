[README.md](https://github.com/user-attachments/files/32812069/README.md)
# **Overview**





This repository contains the main Python 3 code used for particle tracking and wave analysis.



For particle tracking, there are three coupled Python files that together implement the GUI application. The application is designed to analyse AVI videos with a bright background and dark particles. Multiple-particle tracking is also supported.



For wave analysis, the workflow consists of several steps:



* PIV (Particle Image Velocimetry)\*\* is used to extract the velocity field for high-concentration particle systems. Based on this information, the relevant velocity parameters are determined for the subsequent analysis. Since, in our case, particle motion is driven by a circular light pattern, we focus on the radial velocity and total velocity.



* Propagation analysis\*\* is performed based on the PIV results. Kymographs of velocity as a function of radial position and time are analysed to characterise the propagation behaviour of the particles. Further analyses are also performed, including determining the frequency of wave recurrence and investigating how waves propagate from the light pattern outward.



* Statistical analysis\*\* is performed for each wavelength using a defined sample size to investigate the tunability of the wavelength. The responses at different wavelengths are compared and characterised using the RMS radial velocity. The corresponding velocity fields are obtained using the optical flow technique.





&#x20;

