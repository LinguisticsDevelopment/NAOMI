package common

const ( //The Default Constant Thought-Types
	INCLUSION int = iota //"and" means two states at once
	EXCLUSION            //"or" means one state or another as options that are mutually exclusive
	CHANGE               //Statements indicating a change from state to state
	STATE                //"is" statements: simply elaborates on another idea
	CONDITION            //"if" means a condition state must be satisfied for another state to happen
	CONCEPT              //Single word to a thought
	CONNECTION
)

const ( //The Default Parts of a Thought
	//Inclusion
	COMPONENT int = iota
	//Exclusion
	POSSIBILITIES
	//Change
	SUBJECT
	OBJECT
	INDIRECT
	//State
	COMPLEMENT
	//Condition
	PARAMETER
	IF
	ELSE
	//Concept
	MULTIPLIERS
	ADDITIVES
	//Connection
	AFFECTED
	RELATED
)

type Ident int

type Thought struct {
	ID          Ident        //used to reference the other thoughts inside the thoughts
	Form        int          //Determines how this thought is used by other thoughts
	Aspects     [][]*Thought //Different forms of thoughts have different keyed usages of the other thoughts referenced inside the lists
	Parameter   *Thought
	Alternative *Thought
	Negated     bool
	Concept     Idea
}
