from roboflow import Roboflow
rf = Roboflow(api_key="vKdHE6IaxwK0VZAG8TdI")
project = rf.workspace("chu-cm-minh").project("gemstone-iysjs")
version = project.version(1)
dataset = version.download("folder")
