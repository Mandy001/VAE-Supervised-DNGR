#coding=utf-8
from sklearn import manifold
from matplotlib import pyplot as plt

def parse_line(line):
	items= line.split()
	id = items[0]
	embedding = [float(item) for item in items[1:]]
	return id, embedding


#vec_all = "wiki_vec_all.txt"
#vec_all = "sdngr_wiki_vec_all.txt"
vec_all = "vaedngr_wiki_vec_all.txt"
#vec_all = "vaesdngr_wiki_vec_all.txt"
vec_label = "../data/wiki/Wiki_category.txt"


with open(vec_all, "r", encoding="utf-8") as f:
	f.readline()
	datas = [parse_line(line) for line in f]


with open(vec_label, "r", encoding="utf-8") as f:
	id_label_dict = {items[0]:int(items[1]) for items in [line.split() for line in f]}

datas = [data for data in datas if id_label_dict[data[0]] < 15]

embeddings = [data[1] for data in datas]
colors = [id_label_dict[data[0]] for data in datas]


tsne = manifold.TSNE(n_components=2, init='random', n_iter=2000, angle=0.5)
# tsne = manifold.TSNE(n_components=2, init='random', perplexity=5, n_iter=2000)
print("start")
y = tsne.fit_transform(embeddings)
print("end")

fig_1 = plt.figure(figsize=(8, 4))
plt.scatter(y[:,0], y[:,1], c=colors)
plt.show()
# fig_1.savefig('visualization_1layers.pdf')
# fig_1.savefig('visualization_2layers.pdf')
# fig_1.savefig('visualization_3layers.pdf')
# fig_1.savefig('visualization_4layers.pdf')
# fig_1.savefig('visualization_5layers.pdf')
# fig_1.savefig('visualization_1layers_5classes.pdf')
# fig_1.savefig('visualization_1layers_3d.pdf')
#fig_1.savefig('visualization_1layers_15classes.pdf')

#fig_1.savefig('dngr.pdf')
#fig_1.savefig('sdngr.pdf')
fig_1.savefig('vaedngr.pdf')
#fig_1.savefig('vaesdngr.pdf')
