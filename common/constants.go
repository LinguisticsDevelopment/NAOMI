package common

const ( //TYPES of thoughts
	NULL        int = iota
	CONJUNCTION     //"And"
	DISJUNCTION     //"Or"
	DEVELOPMENT     //Verb Phrases
	CONDITION       //"be"
	IMPLICATION     //"if..then"
	EQUIVALENCE     //"if and only if"

	//Concepts:
	ADV
	ADJ
	NOUN
	VERBAL
)

const ( //ASPECTS of thoughts
	//Inclusion
	COMPONENT int = iota

	//Exclusion
	POSSIBILITY

	//Change and State
	SUBJECT
	OBJECT
	INDIRECT
	COMPLEMENT
	MODIFIER

	//Condition
	PARAMETER
	IF
	ELSE

	//Concept
	MULTIPLIER
	ADDITIVE

	//Connection
	AFFECTED
	RELATED
)

const ( //SUBTYPES of thoughts ACROSS ALL LANGUAGES
	COMPARER int = iota
	SUPERLATIVE

	DEGREE

	PRONOUN

	PLURAL
	NEGATIVE

	DETERMINER
	AUXILARY
)
