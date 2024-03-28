package main

func main() {
	likes := NewNode(N_VERBAL, "like", POS_VERB)
	boy := NewNode(N_NOMINAL, "boy", POS_NOUN)
	the := NewNode(N_DESCRIPTOR, "the", POS_DET)
	tall := NewNode(N_DESCRIPTOR, "tall", POS_DET)
	and := NewNode(N_NOMINAL, "and", POS_CCONJ)
	trucks := NewNode(N_NOMINAL, "truck", POS_NOUN)
	cars := NewNode(N_NOMINAL, "car", POS_NOUN)
	trains := NewNode(N_NOMINAL, "train", POS_NOUN)

	//A is the Upper Node

	Connect(C_SUBJECT, likes, boy)

	Connect(C_DESCRIPTION, boy, the)
	Connect(C_DESCRIPTION, boy, tall)

	Connect(C_OBJECT, likes, and)

	Connect(C_COORDINATION, and, trucks)
	Connect(C_COORDINATION, and, cars)
	Connect(C_COORDINATION, and, trains)

	root := likes
	graph := Web{nil, root}

	PrintGraph(graph)
}
