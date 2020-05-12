import javalang as jl
import pandas as pd
import statistics as stat
import os
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import as_float_array
from sklearn.metrics import silhouette_score

inputPath = "PATH_TO/xerces2-j"
df = []
counter = 0
for path, dirs, files in os.walk(inputPath):
    for name in files:
        if name.endswith(".java"):
            finalPath = str(path) + "/" + str(name)
            sc = open(finalPath, 'r').read()
            sc.replace('\n', '')
            tree = jl.parse.parse(sc)
            for root, c in tree.filter(jl.tree.ClassDeclaration):
                for root1, m in c.filter(jl.tree.MethodDeclaration):
                    counter += 1
                df.append({"File_Name": name, "Class_Name": c.name, "Method_Number": counter})
                counter = 0

df = pd.DataFrame(df)

mean = stat.mean(df["Method_Number"].values)
std = stat.stdev(df["Method_Number"].values)

df["Is_God_Class"] = df["Method_Number"].apply(lambda x: "True" if x > (mean + (6 * std)) else "False")

# Feature Vectore



inputPath = "PATH_TO/xerces2-j"


def getFields(Class):
    l = []
    for root, v in Class.filter(jl.tree.VariableDeclarator):
        l.append(v.name)
    for root, m in Class.filter(jl.tree.MethodDeclaration):
        l.append(m.name)
    return l


def getMethods(Class):
    l = []
    for root, m in Class.filter(jl.tree.MethodDeclaration):
        if not m.name in l:
            l.append(m.name)
    return l


def getAccessedFieldsAndMethodsByMethods(Method):
    l = []
    for root, am in Method.filter(jl.tree.MemberReference):
        l.append(am.member)
    for root, am in Method.filter(jl.tree.MethodInvocation):
        l.append(am.member)
    return l


GodClasses = df[df["Is_God_Class"] == "True"].File_Name.tolist()

GodClasses = [x[:-5] for x in GodClasses]

df1 = pd.DataFrame([])
for path, dirs, files in os.walk(inputPath):
    for name in files:
        if name.endswith(".java"):
            finalPath = str(path) + "/" + str(name)
            sc = open(finalPath, 'r').read()
            sc.replace('\n', '')
            tree = jl.parse.parse(sc)
            for root, c in tree.filter(jl.tree.ClassDeclaration):
                if c.name in GodClasses:
                    df1 = pd.DataFrame(columns=getFields(c), index=getMethods(c))
                    l2 = getFields(c)
                    for root1, m in tree.filter(jl.tree.MethodDeclaration):
                        l1 = getAccessedFieldsAndMethodsByMethods(m)
                        l3 = []
                        for x in l2:
                            if x in l1:
                                l3.append(1)
                            else:
                                l3.append(0)
                        df1.loc[m.name] = l3
                        l3 = []
                    df1.to_csv('PATH_TO/FeatureVectors/' + c.name + '.csv')
# Clustering
inputPath = "PATH_TO/FeatureVectors/"
for path, dirs, file in os.walk(inputPath):
    for name in file:
        if name.endswith(".csv"):
            with open(path + "/" + name, newline='') as csvfile:
                rawData = list(csv.reader(csvfile))
            df = pd.read_csv(path + "/" + name)
            data = [x[1:] for x in rawData[1:]]
            data = [as_float_array(x[1:]) for x in rawData[1:]]

            kmax = 0
            kchosenK = 0
            for i in range(2, 40):
                kmeans = KMeans(n_clusters=i).fit(data)
                score = silhouette_score(data, kmeans.labels_)
                if (kmax < score):
                    kmax = score
                    kchosenK = i

            hmax = 0
            hchosenK = 0
            for i in range(2, 40):
                model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='complete').fit(data)
                score = silhouette_score(data, model.labels_)
                if (hmax < score):
                    hmax = score
                    hchosenK = i

            if (kmax > hmax):
                kmeans = KMeans(n_clusters=kchosenK).fit(data)
                finalDf = {"Cluster_ID": kmeans.labels_.tolist(), "Method_Name": df['Unnamed: 0'].tolist()}
                finalDf = pd.DataFrame(finalDf)
                finalDf.to_csv('PATH_TO/Cluster/KMeansCLuster_' + name)
            else:
                model = AgglomerativeClustering(n_clusters=hchosenK, affinity='euclidean', linkage='complete').fit(data)
                finalDf = {"Cluster_ID": model.labels_.tolist(), "Method_Name": df['Unnamed: 0'].tolist()}
                finalDf = pd.DataFrame(finalDf)
                finalDf.to_csv('PATH_TO/Cluster/HierarchicalCluster_' + name)
#Precisions and recall


path = "PATH_TO/Cluster/"

for path, dirs, file in os.walk(path):
    for name in file:
        if name.endswith(".csv"):
            df = pd.read_csv(path + "/" + name)

            f = open('PATH_TO/keywords.txt', 'r')

            list_of_lists = []
            for line in f:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                list_of_lists.extend(line_list)


            def func(x):
                for item in list_of_lists:
                    if item in x:
                        return item
                    else:
                        return 'other'


            df['Cluster'] = df['Method_Name'].apply(func)

            df = df.groupby("Cluster")['Method_Name'].apply(lambda tags: ','.join(tags))

            groundTruth = []
            for x in df:
                groundTruth.append(x.split(","))

            df = pd.read_csv(path + "/" + name)

            df = df.groupby("Cluster_ID")['Method_Name'].apply(lambda tags: ','.join(tags))

            ClusteringGroupBy = []
            for x in df:
                ClusteringGroupBy.append(x.split(","))

            def intra_pairs(Clust):
                intraPairs = set()
                for i in range(len(Clust)):
                    for j in range(len(Clust[i])):
                        for k in range(len(Clust[i])):
                            if j < k:
                                intraPairs.add("<" + Clust[i][j] + "," + Clust[i][k] + ">")
                return intraPairs

            intraC = intra_pairs(ClusteringGroupBy)
            intraG = intra_pairs(groundTruth)
            prec = len(intraC.intersection(intraG)) / len(intraC)
            recall = len(intraC.intersection(intraG)) / len(intraG)
            print("ClassName:" , name , " | P =" , prec , " | R =" , recall , "| F1" , 2 * prec * recall / prec + recall)
