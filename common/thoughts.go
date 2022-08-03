package common

type Id int

type Connect struct {
	Receiver   Id
	Connector  Id
	Referenced Id
}

type Thought struct {
	Part        string
	Form        int          //Determines how this thought is used by other thoughts
	RepForm     int          //for Inclusion and Exclusion, as well as Verbals
	ID          Id           //used to reference the other thoughts inside the thoughts
	Aspects     [][]*Thought //Different forms of thoughts have different keyed usages of the other thoughts referenced inside the lists
	Parameter   Id
	Alternative Id
	Connections []Connect
	Negated     bool
	Consumed    bool
	Concept     Idea
}
