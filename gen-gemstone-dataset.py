from roboflow import Roboflow
rf = Roboflow(api_key="vKdHE6IaxwK0VZAG8TdI")
project = rf.workspace("fma04-fayoum-edu-eg").project("gemstone-classification")
version = project.version(3)
dataset = version.download("folder")