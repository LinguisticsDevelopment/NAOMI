package words

type Word struct {
	//Names
	Name string
	Part string

	//Job Description
	ObjType  string   //adj adv noun verb prep coord suboord
	Function string   //what Function the word is used as if it is changeable
	SubType  string   //for Adj/Adv: norm, compar, super; for Verb: types of verbs
	prepFunc []string //What axises the prepositions effect

	//Execution Order
	Level int //What depth level the word is on

	//Parsing Tags
	Consumed      bool
	Suboordinated bool

	//Producers
	Nominals map[string]Meaning
	Scopes   map[string]Meaning
	Roles    map[string]Meaning

	//Facilitators
	Subject           *Word
	SubjectComplement *Word
	Objects           []*Word
	//IndirectObject    *Word

	//Administrators
	Descriptors []*Word //Adjectives, Numerals, and Determiners - Nominals
	Specifiers  []*Word //Adverbs, Negators & Affirmers - Scopes
	Modifiers   []*Word //Modals - Roles

	//Directors
	Connections map[*Word]*Word //Affect Scope and or Verbs - Relative Connections
	Connected   *Word           //What word a Preposition is tying to
	Relater     *Word           //What word a Preposition is representing

	//Representatives
	Complements     []*Word //Hold Words inside of Coordinators or Suboordinators
	Suboordinations []*Word //Hold Suboordinative Clauses for Words they affect

}

func BuildWord(wordName string, wordPart string) Word {
	defaultNominals := []string{"determinatory", "personal", "living", "permanence", "embodiment", "magnitude"}
	scopesList := []string{"temporal", "frequency", "location", "manner", "extent", "reason", "attitude", "relative", "direction", "spacialExtent", "beneficiary"}
	defaultScopes := []string{}
	roleList := []string{"fundemental", "subject", "subjectComp", "objects", "results", "instruments", "sources", "goals", "experiencer", "nominal"}
	defaultRoles := []string{}

	nominals := map[string]Meaning{}
	nominals["fundemental"] = BuildMeaning("fundemental", true, defaultNominals, 0.0)
	nominals["attributes"] = BuildMeaning("attributes", true, []string{}, 0.0)

	scopes := map[string]Meaning{}
	for _, scopeName := range scopesList {
		scopes[scopeName] = BuildMeaning(scopeName, true, defaultScopes, 1.0)
	}

	roles := map[string]Meaning{}
	for _, roleName := range roleList {
		roles[roleName] = BuildMeaning(roleName, true, defaultRoles, 0.0)
	}
	newWord := Word{Name: wordName, Part: wordPart, Function: wordPart, Nominals: nominals, Scopes: scopes, Roles: roles}

	newWord.Subject = &Word{}
	newWord.SubjectComplement = &Word{}
	newWord.Objects = []*Word{}

	newWord.Specifiers = []*Word{}
	newWord.Descriptors = []*Word{}
	newWord.Modifiers = []*Word{}

	newWord.Connections = map[*Word]*Word{}

	newWord.Complements = []*Word{}

	return newWord
}

//maybe eliminate the list in meaning obj because it will never be multiple words they will be stored in and/onjunction
