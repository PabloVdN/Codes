These three codes produce the figures contained in their names from the article "Non-resonant spin injection of exciton-polaritons with halide perovskites at room temperature" from the authors Pablo Vaquer de Nieves, Elena Sendarrubias Arias-Camisón, Jorge Cuadra, Maksim Lednev, Raúl Gago, Luis Viña, Francisco José García Vidal, Johannes Feist, Ferry Prins and Carlos Antón Solanas.

To produce these figures, the actual codes are built to work using the "Measurements" folder from https://edatos.consorciomadrono.es/dataset.xhtml;jsessionid=50b84d06adafeedbe0e66044f498?persistentId=doi%3A10.21950%2F2KNFQR&version=DRAFT.
These codes work by retrieving metadata from the measurements in this way: 1) Take a step back from a folder where the three codes are contained. 2) Move towards the folder "Measurements". 3) Move towards the folders inside "Measurements", which are "Calibration_wavelength", "Calibration_white_light", "PL_reflec", "S3_exciton" and "S3_polariton", as convenient. These five folders contain all the necessary metadata files to produce Fig. 2,3,4 and Suppl. 4,5,6,7,8.

a) "Calibration_wavelength" contains the files from which the wavelength <-> pixel relation is found for all the measurements.

b) "Calibration_white_light" contains the files that account for the correction of the white light reflectivity measurements, as the white light intensity is not constant in wavelength (or energy).

c) "PL_reflec" contains the files of photoluminescence and white light reflectivity from the three different exciton-polariton detunings from Fig.2.

d) "S3_exciton" contains the files of photoluminescence polarization tomography for the PEPI exciton from Fig. Suppl. 8.

e) "S3_polariton" contains the files of photoluminescence polarization tomography for three exciton-polariton detunings from Fig. 3, 4 and Suppl. 4, 5, 6, 7.

More information can be found in the comments within the code.