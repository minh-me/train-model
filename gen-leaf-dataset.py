from roboflow import Roboflow
rf = Roboflow(api_key="vKdHE6IaxwK0VZAG8TdI")
project = rf.workspace("chu-cm-minh").project("leaf-classification-ozr81")
version = project.version(1)
dataset = version.download("folder")