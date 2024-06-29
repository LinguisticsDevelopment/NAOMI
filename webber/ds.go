package main

import (
	"os"
	"unicode"

	"github.com/google/uuid"
	"github.com/jdkato/prose/v2"

	"bytes"
	"fmt"
	"log"

	"github.com/goccy/go-graphviz"
)

type Word struct {
	Text     string
	POS      Tag
	SubTypes []SubType
}

type Node struct {
	Type  NodeType
	OG    NodeType
	Value *Word
	POS   Tag
	Flags []SubType

	Connections []*Connection
}

func NewNode(tp NodeType, text string, POS Tag) *Node {
	return &Node{tp, tp, &Word{text, POS, []SubType{}}, POS, []SubType{}, []*Connection{}}
}

func (n *Node) AddConnection(connection *Connection) {
	n.Connections = append(n.Connections, connection)
}

func NodesEqual(a *Node, b *Node) bool {
	if a.Type != b.Type {
		return false
	}

	if a.Value.Text != b.Value.Text {
		return false
	}

	if a.POS != b.POS {
		return false
	}

	matches := 0
	for _, ac := range a.Connections {
		for _, bc := range b.Connections {
			if ConnectionsEqual(ac, bc) {
				matches += 1
			}
		}
	}

	if matches != len(a.Connections) {
		return false
	}

	return true
}

type Connection struct {
	Type ConnectionType

	A *Node
	B *Node
	//Nodes []*Node
}

func NewConnection(tp ConnectionType, a *Node, b *Node) *Connection {
	return &Connection{tp, a, b}
}

func Connect(tp ConnectionType, a *Node, b *Node) *Connection {
	c1 := NewConnection(tp, a, b)
	a.AddConnection(c1)
	b.AddConnection(c1)

	return c1
}

func ConnectionsEqual(a *Connection, b *Connection) bool {
	if a.Type != b.Type {
		return false
	}

	// TODO: make sure that A -> B for both
	if a.A.Value != b.A.Value {
		return false
	}

	if a.B.Value != b.B.Value {
		return false
	}

	return true
}

type Web struct {
	// TODO: this should be nuked maybe???? idk???
	// lol
	Sentence []*Word
	Root     *Node
}

func NewWeb(sentence string) (Web, error) {
	doc, _ := prose.NewDocument(sentence)

	var words []*Word

	for i, tok := range doc.Tokens() {
		// l, _, _ := Lem.Lemma(tok.Text, tok.Tag)
		l := tok.Text
		l += fmt.Sprintf(" (%d)", i)
		fmt.Fprintf(os.Stderr, "%s %s\n", tok.Text, tok.Tag)

		if !unicode.IsPunct([]rune(tok.Tag)[0]) {
			newWord := &Word{l, PennToUniv(tok.Tag), []SubType{}}
			if newWord.Text == "what" {
				newWord.SubTypes = append(newWord.SubTypes, S_QUESTION)
			}
			words = append(words, newWord)
		}
	}

	root, err := Parse(words)
	if err != nil {
		return Web{words, nil}, err
	}

	return Web{words, root}, nil
}

func Parse(sentence []*Word) (*Node, error) {
	ParserInit()

	nodes := make([]*Node, 0, len(sentence))
	for _, word := range sentence {
		if word.Text == "either" {
			nodes = append(nodes, &Node{TagToNodeType[word.POS], TagToNodeType[word.POS], word, word.POS, word.SubTypes, []*Connection{}}) //TODO: bad hack
		}
		nodes = append(nodes, &Node{TagToNodeType[word.POS], TagToNodeType[word.POS], word, word.POS, word.SubTypes, []*Connection{}})
	}

	root, err := SentenceParse(nodes)
	return root, err
}

func PrintWeb(web Web) {
	g := graphviz.New()
	graph, _ := g.Graph()

	defer func() {
		if err := graph.Close(); err != nil {
			log.Fatal(err)
		}
		g.Close()
	}()

	root := web.Root

	var AddNode func(root *Node, parent *Node)
	AddNode = func(root *Node, parent *Node) {
		n, _ := graph.CreateNode(root.Value.Text)
		// n, _ := graph.CreateNode(uuid.New().String())
		// n.SetLabel(root.Value.Text)
		for _, connection := range root.Connections {
			if connection.A != root && connection.A != parent {
				m, _ := graph.CreateNode(connection.A.Value.Text)
				// m, _ := graph.CreateNode(uuid.New().String())
				// m.SetLabel(connection.A.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString[connection.Type])
				AddNode(connection.A, root)
			}
			if connection.B != root && connection.B != parent {
				m, _ := graph.CreateNode(connection.B.Value.Text)
				// m, _ := graph.CreateNode(uuid.New().String())
				// m.SetLabel(connection.B.Value.Text)
				e, _ := graph.CreateEdge(uuid.New().String(), n, m)
				e.SetLabel(ConnectionTypeToString[connection.Type])
				AddNode(connection.B, root)
			}
		}
	}
	AddNode(root, nil)

	// n, _ := graph.CreateNode("n")
	// m, _ := graph.CreateNode("m")
	// e, _ := graph.CreateEdge("e", n, m)
	// e.SetLabel("e")

	var buf bytes.Buffer
	if err := g.Render(graph, "dot", &buf); err != nil {
		log.Fatal(err)
	}
	fmt.Println(buf.String())
}

func SimplePrintWeb(web *Web) {
	var drill func(node *Node)
	drill = func(node *Node) {
		fmt.Fprintf(os.Stderr, "(%s, %s),", node.Value.Text, TagToString[node.POS])
		for _, c := range node.Connections {
			if c.B.Value == node.Value {
				continue
			}
			drill(c.B)
		}
	}

	drill(web.Root)
	fmt.Println()
}

func DeepCopyNode(n *Node, into *Node) *Node {
	var newNode *Node = nil
	if into == nil {
		newNode = &Node{n.Type, n.Type, n.Value, n.POS, n.Flags, []*Connection{}}
	} else {
		newNode = into
	}

	for _, c := range n.Connections {
		skip := false
		for _, o := range newNode.Connections {
			if ConnectionsEqual(c, o) {
				skip = true
			}
		}

		if skip {
			continue
		}

		nextNode := &Node{c.B.Type, c.B.Type, c.B.Value, c.B.POS, c.B.Flags, []*Connection{}}

		newConn := &Connection{c.Type, newNode, nextNode}

		newNode.Connections = append(newNode.Connections, newConn)
		nextNode.Connections = append(nextNode.Connections, newConn)

		DeepCopyNode(c.B, nextNode)
	}

	return newNode
}

func DeepCopyWeb(w *Web) *Web {
	oldRoot := w.Root

	root := DeepCopyNode(oldRoot, nil)

	return &Web{w.Sentence, root}
}

func findNodeOfType(root *Node, tp Tag) *Node {
	if root.Value.POS == POS_CCONJ {
		return root
	}

	for _, c := range root.Connections {
		res := findNodeOfType(c.B, tp)
		if res != nil {
			return c.B
		}
	}

	return nil
}

func SplitWebAtCoord(w Web) []*Web {
	res := []*Web{}

	node := findNodeOfType(w.Root, POS_CCONJ)

	for i := range len(node.Connections) - 1 {
		copy := DeepCopyWeb(&w)
		cnode := findNodeOfType(copy.Root, POS_CCONJ)

		var parentConn *Connection = nil
		for _, c := range cnode.Connections {
			if c.B == cnode {
				parentConn = c
			}
		}

		correctCons := []*Connection{parentConn}
		k := i
		for _, c := range cnode.Connections {
			if c == parentConn {
				continue
			}
			if k > 0 {
				k -= 1
				continue
			}

			parentConn.B = c.B
			c.A = parentConn.A
			correctCons = append(correctCons, c)
			break
		}
		cnode.Connections = correctCons

		res = append(res, copy)
	}

	return res
}

func compareWebInternal(a *Node, b *Node, visited *[]*Node) bool {
	for _, node := range *visited {
		if NodesEqual(a, node) {
			return true
		}

		if NodesEqual(b, node) {
			return true
		}
	}

	if a.Value.Text != b.Value.Text {
		return false
	}

	*visited = append(*visited, a)
	*visited = append(*visited, b)

	for _, ac := range a.Connections {
		for _, bc := range b.Connections {
			// if ConnectionsEqual(ac, bc) {
			if ac.Type == bc.Type {
				res := compareWebInternal(ac.B, bc.B, visited)

				if !res {
					return false
				}
			}
		}
	}

	return true
}

func (w *Web) CompareWeb(context *Web) bool {
	visited := []*Node{}
	return compareWebInternal(w.Root, context.Root, &visited)
}

func (w *Web) CompareWebs(context []*Web) bool {
	for _, c := range context {
		fmt.Fprintf(os.Stderr, "Comparing ")
		SimplePrintWeb(w)
		fmt.Fprintf(os.Stderr, " with     ")
		SimplePrintWeb(c)

		res := w.CompareWeb(c)
		if res {
			SimplePrintWeb(c)
			return true
		}
	}

	return false
}
