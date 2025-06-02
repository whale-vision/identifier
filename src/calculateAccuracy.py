import csv
import os
import random
import numpy as np
from tqdm import tqdm
import json
import concurrent.futures

from featureExtraction.featureExtraction import extractor
from segmentation.segmentation import segmentor



HOLDOUT_SET = [
	# dd
	'088_TITTI', '032_ZAZA', '060_LAGAVULIN', '020_ANGELA (female)', '047_SABELLA (female)', '042_PASQUALE', '037_FRODO', '039_SALVATORE', '004_BRAMANTE', '074_BELINDA',
	
    # gg
	'Gg_ODO019_GOYA_ROLLO', 'Gg_ODO014_CARAVAGGIO_SNOOPY', 'Gg_OD011_MANET_ARYA', 'Gg_ODO047_LIEBERMAN',
	
    # pm
	'PMOD032_ELESSAR', 'PMOD021_HAGALAZ', 'PMOD110_FOURIER', 'PMOD093_NOAH', 'PMOD004_ZORA', 'PMOD095_TYRION', 'PMOD100_SANDOR', 'PMOD096_EDDARD', 'PMOD056_NIHAL', 'PMOD024_SIRIO', 'PMOD040_JOSHUA',

    # tt
	'Tt_ODO0143_IDP_ANTEROS_JUV', 'Tt_ODO0088_IDP_PODARKE', 'Tt_ODO0093_IDP_ALKE', 'Tt_ODO0108_IDP_EGEONE_FERITA TESTA', 'Tt_ODO0151_IDP_NEIKEA', 'Tt_ODO0048_IDP_GRATION', 'Tt_ODO0016_IDP_IPERIONE', 'Tt_ODO0114_IDP_DEINO', 'Tt_ODO0202_IDP_ALGEA', 'Tt_ODO0030_IDP_HELIOS', 'Tt_ODO0105_IDP_DEIMOS', 'Tt_ODO0064_IDP_STEROPE', 'Tt_ODO0172_IDP_KAKIA', 'Tt_ODO0072_IDP_GIE', 'Tt_ODO0121_IDP_HYDROS_F', 'Tt_ODO0189_IDP_EMPUSA', 'Tt_ODO0176_IDP_HYBRIS', 'Tt_ODO0055_IDP_ACLI', 'Tt_ODO0175_IDP_HYPNOS', 'Tt_ODO0139_IDP_FOBETORE'
]




def getAllWhales(directory):
    whaleDict = {}

    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        # Get the relative path from the root directory
        relativePath = os.path.relpath(root, directory)
        # Split the relative path to find the first subdirectory (whale name)
        pathParts = relativePath.split(os.sep)

        if len(pathParts) > 0 and pathParts[0] != ".":
            whaleName = pathParts[0]  # First subdirectory is the whale name

            # Initialize the list for this whale if it doesn't exist
            if whaleName not in whaleDict:
                whaleDict[whaleName] = []

            # Add each file to the whale's list
            for file in files:
                fullPath = os.path.join(root, file)
                whaleDict[whaleName].append((whaleName, fullPath))

    return whaleDict





# create dataset
def createDataset(modelPath, directory):
	segmentationModel = segmentor()
	featureModel = extractor(modelPath, modelPath)

	allWhales = getAllWhales(directory)

	whaleIdentities = {}

	for whale in tqdm(allWhales):

		segmented = [segmentationModel.crop(imageData[1]) for imageData in allWhales[whale]]
		segmented = [whale for whale in segmented if whale is not None]

		whaleIdentities[whale] = []

		for imageData in segmented:
			imageData.update({"identity": whale})

			embedding = featureModel.extract(imageData)

			whaleIdentities[whale].append(embedding)

	return whaleIdentities

# split between reference and test
def splitDataset(whaleIdentities, referenceSize):
	reference = {}
	test = {}

	for whale in whaleIdentities:
		if len(whaleIdentities[whale]) <= referenceSize + 1:
			continue

		reference[whale] = []
		test[whale] = []

		random.shuffle(whaleIdentities[whale])

		for i in range(referenceSize):
			reference[whale].append(whaleIdentities[whale][i])

		for i in range(referenceSize, len(whaleIdentities[whale])):
			test[whale].append(whaleIdentities[whale][i])

	return reference, test


# create reference averages
def calculateAverages(references):
	averages = {}

	for whale in references:
		average = np.mean(references[whale], axis=0)
		averages[whale] = average

	return averages


# create distances
def calculateDistances(references, test):
	distances = {}

	for whale in test:
		whaleDistances = []

		for testEmbedding in test[whale]:
			embeddingDistances = []

			for referenceWhale in references:
				distance = np.linalg.norm(testEmbedding - references[referenceWhale])
				embeddingDistances.append((referenceWhale, distance))

			whaleDistances.append(embeddingDistances)
		distances[whale] = whaleDistances

	return distances


# calculate whale accuracy
def calculateWhaleAccuracy(distances, topN):
	accuracies = []
	holdoutAccuracies = []

	for whale in distances:
		whaleAccuracies = []
		whaleHoldoutAccuracies = []

		for imageEmbeddings in distances[whale]:
			imageEmbeddings.sort(key=lambda x: x[1])

			topNResults = [result[0] for result in imageEmbeddings[0:topN]]

			accuracy = 1 if whale in topNResults else 0

			if whale in HOLDOUT_SET:
				whaleAccuracies.append(accuracy)

			else:
				whaleHoldoutAccuracies.append(accuracy)
		
		if len(whaleAccuracies): accuracies.append(np.mean(whaleAccuracies))
		if len(whaleHoldoutAccuracies): holdoutAccuracies.append(np.mean(whaleHoldoutAccuracies))

	average_accuracy = np.mean(accuracies)
	holdout_accuracy = np.mean(holdoutAccuracies)

	return average_accuracy, holdout_accuracy


def processIteration(args):
	whaleIdentities, referenceSize, topN = args

	reference, test = splitDataset(whaleIdentities, referenceSize)

	references = calculateAverages(reference)
	distances = calculateDistances(references, test)

	accuracy, holdoutAccuracy = calculateWhaleAccuracy(distances, topN)

	return (accuracy, holdoutAccuracy)


# calculate total accuracy
def calculateAccuracy(modelPath, directory, outputPath):
	whaleIdentities = createDataset(modelPath, directory)

	for whale in whaleIdentities:
		if whale is None:
			continue
		
		with open(outputPath, "a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow([whale, "", whaleIdentities[whale]])

	topNAccuracies = {}
	topNHoldoutAccuracies = {}

	for topN in [1, 5]:
		averageAccuracies = []
		averageHoldoutAccuracies = []

		for referenceSize in tqdm(range(5, 6)):
			accuracies = []
			holdoutAccuracies = []
			with concurrent.futures.ProcessPoolExecutor() as executor:
				# do it 100 times, in parallel
				results = executor.map(processIteration, [(whaleIdentities, referenceSize, topN)] * 1)
				results = list(results)

				for result in results:
					accuracy, holdoutAccuracy = result

					accuracies.append(accuracy)
					holdoutAccuracies.append(holdoutAccuracy)

			average_accuracy = np.mean(accuracies)
			average_holdout_accuracy = np.mean(holdoutAccuracies)

			averageAccuracies.append(average_accuracy)
			averageHoldoutAccuracies.append(average_holdout_accuracy)

		topNAccuracies[topN] = averageAccuracies
		topNHoldoutAccuracies[topN] = averageHoldoutAccuracies

	return topNAccuracies, topNHoldoutAccuracies


def runAccuracyTest():
	whaleSets = [
		# {
		# 	"species": "dd",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\dd",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\dd.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\test_models\\dd_identification_flank_ind.csv"
		# },
		# {
		# 	"species": "gg",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\gg",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\gg.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\test_models\\gg_identification_flank_ind.csv"
		# },
		# {
		# 	"species": "tt",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\tt",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\tt.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\test_models\\tt_identification_flank_ind.csv"
		# },
		# {
		# 	"species": "pm_fluke",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\pm\\fluke",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_fluke.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_identification_fluke_ind.csv"
		# },
		{
			"species": "pm_flank",
			"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\pm\\flank",
			"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_flank.pth",
			"outputPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_identification_flank_ind.csv"
		},
		# {
		# 	"species": "dd_pm",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\dd",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_flank.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\models\\dd_pm_identification_flank_ind.csv"
		# },
		# {
		# 	"species": "gg_pm",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\gg",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_flank.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\models\\gg_pm_identification_flank_ind.csv"
		# },
		# {
		# 	"species": "tt_pm",
		# 	"dataset": "G:\\Whale Stuff\\_data\\cetaceans\\curated\\tt",
		# 	"networkPath": "G:\\Whale Stuff\\Identifier\\src\\models\\pm_flank.pth",
		# 	"outputPath": "G:\\Whale Stuff\\Identifier\\src\\models\\tt_pm_identification_flank_ind.csv"
		# }
	]

	accuracies = {}

	for whaleSet in whaleSets:
		print(f"Running accuracy test for {whaleSet['species']}")
		
		if os.path.exists(whaleSet["outputPath"]):
			os.remove(whaleSet["outputPath"])

		results = calculateAccuracy(
			whaleSet["networkPath"],
			whaleSet["dataset"],
			whaleSet["outputPath"]
		)

		accuracies[whaleSet["species"]] = results

	print(accuracies)

	with open("accuracies.json", "w") as f:
		json.dump(accuracies, f, indent=4)

if __name__ == "__main__":
	runAccuracyTest()
