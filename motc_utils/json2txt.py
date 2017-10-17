import glob, os
import json

def getSeqNameFromPath(path):
  return path.split("/")[7]

with open("dev_paths.txt", encoding="utf-8") as file:
  my_list = file.readlines()
  all_dirs = [x.strip() for x in my_list]

  for path in all_dirs:
    pathToOut = path + "/out"

    print("Now in " + pathToOut)
    os.chdir(pathToOut)
    files = glob.glob("*.json")
    files = sorted(files)
    result = []
    for file in files:
      pathToFile = pathToOut + "/" + file
      print(pathToFile)
      with open(pathToFile) as json_data:
        data = json.load(json_data)
        result += [file.split(".json")[0].lstrip("0") + ", -1, " + str(object["topleft"]["x"]) + ", " + str(object["topleft"]["y"]) + ", " + str(object["bottomright"]["x"] - object["topleft"]["x"]) + ", " + str(object["bottomright"]["y"] - object["topleft"]["y"]) + ", " + str(object["confidence"]) for object in data]

      thefile = open('/home/frolov/U/' + getSeqNameFromPath(path) + '.txt', 'w')
      for item in result:
        thefile.write("%s\n" % item)
