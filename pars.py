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
                df.append({"File_Name": name, "Class_Name": c.name, "Method_Number": len(c.methods)})

df = pd.DataFrame(df)

mean = stat.mean(df["Method_Number"].values)
std = stat.stdev(df["Method_Number"].values)

df["Is_God_Class"] = df["Method_Number"].apply(lambda x: "True" if x > (mean + (6 * std)) else "False")


# Feature Vectore



inputPath = "PATH_TO/xerces2-j"
def getFields(Class):
    l = []
    for f in Class.fields:
        for x in f.declarators:
            l.append(x.name)
    return l


def getMethods(Class):
    l = []
    objectM = []
    for m in Class.methods:
        l.append(m.name)
        objectM.append(m)
    return list(set(l)), objectM


def getAccessedFieldsAndMethodsByMethods(Method):
    l = []
    for body in m.body:
        if(type(body) is jl.tree.StatementExpression):
            if(type(body.expression) == jl.tree.Assignment):
                l.append(body.expression.expressionl.member)
            elif(type(body.expression) == jl.tree.MethodInvocation):
                l.append(body.expression.member)   
    return l



def getFieldsMethod(method):
    fieldsReference = list(method.filter(jl.tree.MemberReference))
    fieldsbyMethod = []
    for i in fieldsReference:
        fieldsbyMethod.append(i[1].member)
    return fieldsbyMethod


def getMethodsMethod(method):
    methodInvocations = list(method.filter(jl.tree.MethodInvocation))
    methodsbyMethod = []
    for i in methodInvocations:
        methodsbyMethod.append(i[1].member)
    return methodsbyMethod





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
                    fields = getFields(c)
                    methods, objmethod = getMethods(c)
                    methods = list(set(methods))
                    objmethod = list(set(objmethod))
                    df1 = pd.DataFrame(columns=fields + methods, index=methods)
                    df1.index.rename('Method_name', inplace=True)
                    df1 = df1.fillna(0)
                    print(c.name,df1.shape)
                    l2 = fields + methods
                    for m in objmethod:
                        l1 = getFieldsMethod(m) + list(set(getMethodsMethod(m)))
                        for x in l2:
                            if x in l1:
                                df1.loc[m.name][x] = 1 
                    df1 = df1.loc[:, (df1 != 0).any(axis=0)]
                    df1 = df1.reset_index()
                    print(c.name,df1.shape)
                    df1.to_csv('/Users/hassanatwi/Desktop/FeatureVectors/' + c.name + '.csv')
# Clustering
inputPath = "PATH_TO/FeatureVectors/"
Klist = []
Hlist = []
for path, dirs, file in os.walk(inputPath):
    for name in file:
        if name.endswith(".csv"):
            with open(path + "/" + name, newline='') as csvfile:
                rawData = list(csv.reader(csvfile))
            df = pd.read_csv(path + "/" + name)
            data = [as_float_array(x[2:]) for x in rawData[1:]]
    
            kmax = 0
            kchosenK = 0
            li = []
            total = 0
            for i in range(2, 61):
                kmeans = KMeans(n_clusters=i).fit(data)
                TestDfk = {"Cluster_ID": kmeans.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
                TestDfk = pd.DataFrame(TestDfk)
                #for j in range(i):
                    #total += TestDfk[TestDfk["Cluster_ID"] == j].describe().loc['count'][0]
                #li.append([i,TestDfk.describe().loc['mean'][0]])
                total = 0
                score = silhouette_score(data, kmeans.labels_)
                li.append([i,score])
                if (kmax < score):
                    kmax = score
                    kchosenK = i

            hmax = 0
            hchosenK = 0
            Hli = []
            total = 0
            for i in range(2, 61):
                model = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='complete').fit(data)
                TestDfk = {"Cluster_ID": model.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
                TestDfk = pd.DataFrame(TestDfk)
                #for j in range(i):
                    #total += TestDfk[TestDfk["Cluster_ID"] == j].describe().loc['count'][0]
                #Hli.append([i,TestDfk.describe().loc['mean'][0]])
                total = 0
                score = silhouette_score(data, model.labels_)
                Hli.append([i,score])
                if (hmax < score):
                    hmax = score
                    hchosenK = i
            print("Kmeans ", name,kchosenK, kmax)
            print("    ")
            print("AgglomerativeClustering ", name,hchosenK, hmax)
            finalDf = {"Cluster_ID": kmeans.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
            finalDf = pd.DataFrame(finalDf)
            finalDf.to_csv('/Users/hassanatwi/Desktop/Cluster/KMeansCLuster_' + name)
            finalDf = {"Cluster_ID": model.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
            finalDf = pd.DataFrame(finalDf)
            finalDf.to_csv('/Users/hassanatwi/Desktop/Cluster/HierarchicalCluster_' + name)
            if (kmax > hmax):
                kmeans = KMeans(n_clusters=kchosenK).fit(data)
                finalDf = {"Cluster_ID": kmeans.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
                finalDf = pd.DataFrame(finalDf)
                finalDf.to_csv('/Users/hassanatwi/Desktop/Cluster/KMeansCLuster_' + name)
            else:
                model = AgglomerativeClustering(n_clusters=hchosenK, affinity='euclidean', linkage='complete').fit(data)
                finalDf = {"Cluster_ID": model.labels_.tolist(), "Method_Name": df['Method_name'].tolist()}
                finalDf = pd.DataFrame(finalDf)
                finalDf.to_csv('/Users/hassanatwi/Desktop/Cluster/HierarchicalCluster_' + name)
            Klist.append(li)
            Hlist.append(Hli)
            
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
