from featureExtraction.featureExtraction import getPath
import numpy as np
import csv
import ast

FLANK_MODEL = getPath()+"/src/models/pm_identification_flank_ind.csv"
FLUKE_MODEL = getPath()+"/src/models/pm_identification_fluke_ind.csv"

def load_identities(CURRENT_FILE: str):
	with open(CURRENT_FILE, "r", encoding='utf-8', errors='ignore') as file:
		reader = csv.reader(file)
		identities = list(reader)

		namedIdentities = [
			(identity[0], ast.literal_eval(identity[2])) for identity in identities
		]

		# Create a dictionary of all the whales
		allIdentities = {}

		for embedding in namedIdentities:
			if embedding[0] not in allIdentities:
				allIdentities[embedding[0]] = []

			allIdentities[embedding[0]].append(embedding[1])

		return allIdentities
     

def create_average_identities(allIdentities):
    # Create the average embeddings for each whale
    averageIdentities = {}

    for whale in allIdentities:
        averageEmbedding = np.mean(allIdentities[whale], axis=0)
        averageIdentities[whale] = averageEmbedding

    return averageIdentities

class identifier:
    def __init__(self):
        self.identities = {
            "flank": load_identities(FLANK_MODEL),
            "fluke": load_identities(FLUKE_MODEL),
        }

        self.calculateAverages()

    def calculateAverages(self):
        self.averageIdentities = {
            "flank": create_average_identities(self.identities["flank"]),
            "fluke": create_average_identities(self.identities["fluke"]),
        }

    def addIdentity(self, whale):
        identity = whale["identity"]
        type = whale["type"]
        embedding = whale["embedding"]

        print(f"Adding identity {identity} of type {type} with embedding {embedding}")
        if identity not in self.identities[type]:
            self.identities[type][identity] = []

        self.identities[type][identity].append(embedding)


    def identify(self, whale):
        try:
            embedding = whale["embedding"]
            identities = self.averageIdentities[whale["type"]]

            distances = [("Unknown", 1)]

            for name in identities.keys():
                identity = identities[name]

                distance = np.linalg.norm(np.array(identity) - np.array(embedding))
                distances.append((name, distance))

            distances.sort(key=lambda x: x[1])

            return {
                "path": whale["path"],
                "distances": distances,
            }
        
        except Exception as e:
            print("WARNING: Failed to identify image.")
            print(e)


class identityCreator:
    def __init__(self):
        self.identities = {
            "flank": [],
            "fluke": [],
        }

    def addImage(self, whale):
        embedding = whale["embedding"]
        self.identities[whale["type"]].append(embedding)

    def getIdentity(self):
        # get average position
        meanFlankEmbedding = (
            (list(np.mean(self.identities["flank"], axis=0)))
            if len(self.identities["flank"]) > 0
            else []
        )
        meanFlukeEmbedding = (
            (list(np.mean(self.identities["fluke"], axis=0)))
            if len(self.identities["fluke"]) > 0
            else []
        )

        flankDistances = []
        if len(meanFlankEmbedding) > 0:
            for name, identity in self.identities["flank"]:
                distance = np.linalg.norm(
                    np.array(identity) - np.array(meanFlankEmbedding)
                )
                flankDistances.append((name, distance))

        flukeDistances = []
        if len(meanFlukeEmbedding) > 0:
            for name, identity in self.identities["fluke"]:
                distance = np.linalg.norm(
                    np.array(identity) - np.array(meanFlukeEmbedding)
                )
                flukeDistances.append((name, distance))

        # combine distances
        distances = flankDistances
        for distance in flukeDistances:
            for i, (name, d) in enumerate(distances):
                if name == distance[0]:
                    distances[i] = (name, (d + distance[1]) / 2)

        distances.sort(key=lambda x: x[1])

        return {
            "distances": distances,
        }

    def saveIdentity(self, target: str, embeddings, whaleName: str):
        meanEmbedding = [whaleName] + list(np.mean(embeddings, axis=0))

        fileName = f"../Identifier/src/models/identification_{target}.csv"

        with open(fileName, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(meanEmbedding)

    def save(self, whaleName):
        self.saveIdentity("flank", self.identities["flank"], whaleName)
        self.saveIdentity("fluke", self.identities["fluke"], whaleName)
