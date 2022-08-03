package parser

import (
	"common"
)

//Buckets

func ABRI(rules RuleSet, roots []common.Idea) {
	buckets := make([]common.Thought, len(roots))
	order := []string{"ADV", "ADJ", "NOUN", "DEVELOPMENT"}
	for _, activePart := range order {
		for rootIndex, root := range roots {
			if root.Part == activePart {
				buckets[rootIndex] = root.Conceptualize()
			}
		}

		for _, rule := range rules[activePart] {
			activeBuckets := make([]*common.Thought, len(roots))
			for index, bucket := range buckets {
				if !bucket.Consumed {
					activeBuckets[index] = &buckets[index]
				}
			}
			applyRules(rule, activeBuckets, roots)
		}
		coordCheck(buckets, roots)
		conjunctCheck(buckets, roots)
		prepCheck(buckets, roots)
	}
}

func wipeConsumed(dirtyBuckets []*common.Thought) []*common.Thought {
	cleanBuckets := make([]*common.Thought, len(dirtyBuckets))
	for index, dirty := range dirtyBuckets {
		if !dirty.Consumed {
			cleanBuckets[index] = dirtyBuckets[index]
		} else {
			cleanBuckets[index] = &common.Thought{Part: "Consumed"}
		}
	}
	return cleanBuckets
}

func applyRules(rule Rule, buckets []*common.Thought, roots []common.Idea) {
	for index, this := range buckets {
		if this.Part == rule.RootType { //checking if root word is the correct part of speech
			if rule.RootSubType != "" { //checking if field is important
				hasSubtype := false
				for _, subtype := range this.Concept.Subtypes {
					if rule.RootSubType == subtype {
						hasSubtype = true
					}
				}
				if !hasSubtype {
					continue
				}
			}
			//If the current word qualifies:

			failed := false
			partsMap := [][]*common.Thought{}
			for _, part := range rule.Parts {
				if failed {
					break
				}
				workingBucket := buckets[index+part.Direction]
				if workingBucket.Part != part.ModType {
					failed = true
					break
				}
				if part.CaseMatters {
					hasCase := false
					for _, subtype := range this.Concept.Subtypes {
						if part.Case == subtype {
							hasCase = true
						}
					}
					if !hasCase {
						failed = true
						break
					}
				}
				partsMap[part.AspectApplied] = append(partsMap[part.AspectApplied], buckets[index+part.Direction])
				if part.AllUntilGap {
					step := -1
					if part.Direction > 0 {
						step = 1
					}
					for i := index; i >= 0 && i < len(buckets); i = i + step {
						additionalBucket := buckets[index+part.Direction+i]
						if additionalBucket.Part == "Consumed" {
							continue
						}
						if additionalBucket.Part != part.ModType {
							break
						}
						if part.CaseMatters {
							hasCase := false
							for _, subtype := range this.Concept.Subtypes {
								if part.Case == subtype {
									hasCase = true
								}
							}
							if !hasCase {
								break
							}
						}
						partsMap[part.AspectApplied] = append(partsMap[part.AspectApplied], buckets[index+part.Direction+i])
					}
				}
			}
			if failed {
				continue
			}
			dirtyBuckets := applyConcept(partsMap, buckets[index])
			buckets = wipeConsumed(dirtyBuckets)
		}
	}
}

func applyConcept(aspects [][]*common.Thought, applied *common.Thought) []*common.Thought {

}

func coordCheck(buckets []common.Thought, roots []common.Idea) {
	for index, this := range buckets {
		if this.Form != common.NULL {

			if (index != 0) && (index != len(buckets)-1) && (roots[index].Part == "coord") {

			}
		}
	}

}

func conjunctCheck(buckets []common.Thought, roots []common.Idea) {
	for index, this := range buckets {
		if this.Form != common.NULL {

			if (index != 0) && (index != len(buckets)-1) && (roots[index].Part == "conj") {

			}
		}
	}

}

func prepCheck(buckets []common.Thought, roots []common.Idea) {
	for index, this := range buckets {
		if this.Form != common.NULL {

			if (index != 0) && (index != len(buckets)-1) && (roots[index].Part == "prep") {

			}
		}
	}

}

func getAt() {

}
