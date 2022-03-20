package common

type Idea struct {
	Name string

	Part         string
	CurrentUsage string
	Subtypes     []string

	Level int

	Consumed      bool
	Suboordinated bool

	Parts       map[string]Meaning //Nominals, Scopes, and Roles are now all interconnected
	Descriptors map[int][]*Thought
}

func (current *Idea) Conceptualize() Thought {
	newThought := Thought{Form: CONCEPT}
	newThought.Concept = *current
	for key, value := range current.Descriptors {
		newThought.Aspects[key] = value
	}

	return newThought
}
