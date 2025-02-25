from roboflow import Roboflow
rf = Roboflow(api_key="r4eBbuh3cCi97H8V0zol")
project = rf.workspace("new-workspace-qwfwp").project("eurocoins-6ayd3-lkh5f")
version = project.version(1)
dataset = version.download("folder")
