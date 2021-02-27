# About AqSolPred

AqSolPred is an highly accurate solubility prediction model that consists consensus of 3 ML algorithms (Neural Nets, Random Forest, and XGBoost). AqSolPred is developed using a quality-oriented data selection method described in [1] and trained on AqSolDB [2] largest publicly available aqueous solubility dataset.

AqSolPred showed a top-performance (0.348 LogS Mean Absolute Error) on Huuskonen benchmark dataset [3].

![alt text](https://raw.githubusercontent.com/mcsorkun/AqSolPred-web/main/streamlit-aqsolpred.gif)

# AqSolPred Web Version

Currently, web version is running on Streamlit Share (the related repository is inside streamlit folder). 
You can visit from the following URL: https://share.streamlit.io/mcsorkun/aqsolpred-web/main/streamlit/app.py

**aqsolpred web version:** 1.0s (lite version of v1.0 described in the paper with reduced RFs(n_estimators=200,max_depth=10) but the same performance)

If you are using the predictions from AqSolPred on your work, please cite these papers: [1, 2]

**Special thanks:** This web app is developed based on the tutorials and the template of [DataProfessor's repository](https://github.com/dataprofessor/code/tree/master/streamlit/part7). 

**Note:** Main folder was prepared for Heroku deployment, however it passes the Heroku slug size therefore it is not online on heroku.  

**PS:** Check out dockerfile in gcloud folder to know how I installed conda + rdkit on google cloud platform.
                                                                                         
**Contact:** [Murat Cihan Sorkun](https://www.linkedin.com/in/murat-cihan-sorkun/)

# References

[1] Sorkun, M. C., Koelman, J.M.V.A. & Er, S.  (2020). Pushing the limits of solubility prediction via quality-oriented data selection, Research Square, DOI: https://doi.org/10.21203/rs.3.rs-84771/v1.

[2] Sorkun, M. C., Khetan, A., & Er, S. (2019).  [AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds](https://www.nature.com/articles/s41597-019-0151-1). Scientific data, 6(1), 1-8.

[3] Huuskonen, J. Estimation of aqueous solubility for a diverse set of organic compounds based on molecular topology. Journal of Chemical Informationand Computer Sciences 40, 773–777 (2000).
