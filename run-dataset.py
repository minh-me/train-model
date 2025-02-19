
from roboflow import Roboflow
rf = Roboflow(api_key="vKdHE6IaxwK0VZAG8TdI")
project = rf.workspace("tririlili").project("tarigemstones")
version = project.version(1)
dataset = version.download("folder")