This project is Python implementation of our hybrid recommender system. To implement our proposal, we utilize both Spyder IDE and Jupiter Lab over Anaconda Navigator. While investigating code snippets is efficient in Jupiter Lab, complete files are executed effectively in Spyder IDE.
In our hybrid recommender system, content-based and collaborative filtering recommender approaches are applied. In both recommender approaches, whole process can be divided into two phases: first is item-item similarity computation and second is prediction computation.
Implementation consists of seven files as remarked in quotation marks in following text. In “content_based_sims_part1.py” contributions of movie genres and directors are computed over content-based similarity computation. We compute sizes of intersection for countries and actor cases. At the end of each step, we update intersize in an appropriate way. You can access their Python files from “content_based_sims_part2.py” and “content_based_sims_part3.py”, respectively.  
In “content_based_preds_comp.py”, we load this similarity data entitled “intersize_gdca_norm_symmetric.npy” into variable igdca using NumPy.load() method. Then, we compute prediction p_ax according to the formula given for content-based recommender.
As a second building block of our hybrid recommender system implementation, we care about item-based collaborative filtering mechanism proposed by Sarwar et al. (2001). Complete code of this computation can be accessed from the file “collab_filtering_sims_comp.py”. In the file entitled “collab_filtering_preds_comp.py”, we compute prediction q_ax in compliance with the given formula for collaborative filtering recommender.
As a final step of a whole merging two different recommender results, we compute prediction via file "alpha_integrator.py".

Sarwar, B. M., Karypis, G., Konstan, J. A., & Riedl, J. T. (2001). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web (pp. 285-295). ACM. https://doi.org/10.1145/371920.372071 