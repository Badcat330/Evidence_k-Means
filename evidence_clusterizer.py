import random
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from pulp import LpMaximize, LpProblem, lpSum, LpVariable, PULP_CBC_CMD
import numpy as np

class EvidenceKMeans(BaseEstimator):
    """
    K-means algorithm for cluster decomposition of the body of Evidence. 
    Implimentation is based on Alexander Lepskiy paper Cluster Decomposition of the Body of Evidence.

    Parameters
    ----------
    l : int,
        The number of clusters.
    con_threshold : float
        The threshold for max cluster conflict. Value in deapozone [0, 1].
    uncertainty_threshold : float
        The uncertainty threshold for alpha computation. Value in deapozone [0, 1].
    conflict measure : Callable, dafault=None
        Function for computing conflict between body of evidence. If None use Dempster rule.

    Attributes
    ----------


    References
    ----------
    .. [1] A. Lepskiy, "Cluster Decomposition of the Body of Evidence", Belief Functions: Theory and Applications, 163-173, 2022.

    """


    def __init__(self, l:int, con_threshold:float, uncertainty_threshold:float=1, conflict_measure:Callable=None) -> None:
        super().__init__()
        self.l = l
        self.con_threshold = con_threshold
        self.uncertainty_threshold = uncertainty_threshold
        if callable(conflict_measure):
            self.conflict_measure = conflict_measure
        else:
            self.conflict_measure = EvidenceKMeans._dempster_conflict_measure

    @staticmethod
    def _dempster_conflict_measure(f_1, f_2) -> float:
        sum = 0
        for i in f_1:
            for j in f_2:
                if len(i[0] & j[0]) == 0:
                    sum += i[1] * j[1]
        
        return sum

    def get_params(self, deep=True):
        return {
            "l": self.l,
            "con_threshold": self.con_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "conflict_measure": self.conflict_measure
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def plausibility(self, cluster_number):
        pl_in_cluster = []
        for A in self._clusters_values[cluster_number]:
            pl = 0
            for B in self._clusters_values[cluster_number]:
                if len(A[0] & B[0]) != 0:
                    pl += B[1]
            pl_in_cluster.append(pl)
        
        return pl_in_cluster



    def _choose_cluster(self, f):
        min_conf = float('inf')
        cluster_number = -1
        for i, cluster in enumerate(self._clusters_centers):
            conf = self.conflict_measure([f], cluster)
            if min_conf > conf:
                min_conf = conf
                cluster_number = i
        
        return (cluster_number, min_conf)

    def fit(self, X, y=None, verbose:int=0):
        X = check_array(X, dtype=object).tolist()

        # Initial step
        is_need_stabilize = True
        self._clusters_centers = []
        self._clusters_values = [[] for _ in range(self.l)]
        for i in range(self.l):
            new_center = (X[i][0], 1)
            self._clusters_centers.append([new_center])

        while(is_need_stabilize):
            # Redistribute focal elements
            is_changed = False
            new_clusters_values = [[] for _ in range(len(self._clusters_centers))]
            for x in X:
                cluster_number, min_cluster_conf = self._choose_cluster(x)
                if min_cluster_conf > self.con_threshold:
                    new_cluster = [(x[0], 1)]
                    self._clusters_centers.append(new_cluster)
                    new_clusters_values.append([[]])
                    cluster_number = -1
                if x not in self._clusters_values[cluster_number]:
                    new_clusters_values[cluster_number].append(x)
                    is_changed = True

            # If clusters don't change stop algorithm
            if not is_changed:
                is_need_stabilize = False
                continue
            else:
                self._clusters_values = new_clusters_values

            # Count new clusters centers
            for i in range(len(self._clusters_centers)):
                pl_values = self.plausibility(i)
                model = LpProblem(sense=LpMaximize)
                obj_func = []
                condition = []
                condition_mass = []
                for j, pl in enumerate(pl_values):
                    alpha = LpVariable(name="alpha" + str(j), lowBound=0, upBound=1)
                    obj_func.append(alpha * pl)
                    condition.append(alpha * np.log(len(self._clusters_values[i][j])))
                    condition_mass.append(alpha)

                model += lpSum(obj_func)
                model += lpSum(condition) <= self.uncertainty_threshold
                model += lpSum(condition_mass) == 1
                model.solve(PULP_CBC_CMD(msg=False))

                alphas = model.variables()
                new_center = []
                for j, alpha in enumerate(alphas):
                    if alpha.value() == 0:
                        continue
                    new_center.append((self._clusters_values[i][j][0], alpha.value()))

                self._clusters_centers[i] = new_center



