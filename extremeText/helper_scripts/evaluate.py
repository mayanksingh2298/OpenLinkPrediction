# python3 evaluate.py <path to test file with targets> <path of confidence file to evaluate>

# to add filtered ranking
# compute metrics for each divison of data
import sys
from tqdm import tqdm
sys.path.append("../")

def main():
	target_entities = []
	lines = open(sys.argv[1],'r').readlines()

	# get gold
	for line in lines:
		entity = line.split()[0]
		entity = int(entity[9:])
		target_entities.append(entity)

	# get predictions
	predicted_entities = []
	lines = open(sys.argv[2],'r').readlines()
	for line in tqdm(lines, desc="get predicted_entities"):
		line = line.strip().split()
		best_score = -999999999
		best_index = -1
		for i in range(0,len(line),2):
			score = float(line[i+1])
			index = int(line[i][9:])
			if score>best_score:
				best_score = score
				best_index = index
		predicted_entities.append(best_index)

	assert len(target_entities) == len(predicted_entities)
	# compute metrics
	hits1 = 0
	for i in range(len(target_entities)):
		if target_entities[i]==predicted_entities[i]:
			hits1+=1

	print(hits1/len(target_entities))


if __name__ == '__main__':
	main()