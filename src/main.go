//If i get it to maximum high level understanding level - make a unique encryption key for every time a new "mind"
//is created, and then destroy the key - allows for computer to have a memory without having it's mind read, and
//for privacy between user and computer to be maintained

package main

import (
	"NAOMI/src/cores"
	. "NAOMI/src/cores/words"
)

func main() {
	bob := BuildWord("bob", "noun")
	jim := BuildWord("jim", "noun")
	//geoff := BuildWord("geoff", "noun")
	//shirt := BuildWord("shirt", "noun")
	//pants := BuildWord("pants", "noun")
	//sea := BuildWord("sea", "noun")
	//stars := BuildWord("stars", "noun")
	//children := BuildWord("children", "noun")
	//he := BuildWord("he", "noun")
	//she := BuildWord("she", "noun")

	//runs := BuildWord("runs", "verb")
	//are := BuildWord("are", "verb")
	//is := BuildWord("is", "verb")
	kills := BuildWord("kills", "verb")
	//slaps := BuildWord("slaps", "verb")
	//murders := BuildWord("murders", "verb")
	//wears := BuildWord("wears", "verb")
	//sits := BuildWord("sits", "verb")

	//bright := BuildWord("bright", "adj")
	//big := BuildWord("big", "adj")
	//red := BuildWord("red", "adj")
	//sad := BuildWord("sad", "adj")
	//upsetting := BuildWord("upsetting", "adj")
	//worse := BuildWord("worse", "adj")
	//worse.SubType = "comp"
	//the := BuildWord("the", "adj")
	//bad := BuildWord("bad", "adj")
	green := BuildWord("green", "adj")
	//blue := BuildWord("blue", "adj")
	//gay := BuildWord("gay", "adj")

	//very := BuildWord("very", "adv")
	//slightly := BuildWord("slightly", "adv")
	//extremely := BuildWord("extremely", "adv")

	//under := BuildWord("under", "prep")
	//than := BuildWord("than", "prep")
	//and := BuildWord("and", "coord")
	//who := BuildWord("who", "suboord")
	//which := BuildWord("which", "suboord")

	sentence := []Word{green, bob, kills, jim}
	//sentence := []Word{bob, kills, jim, under, the, sea}
	cores.TotalParse(sentence)
}
