package main

var L_AND = Word{"&&", POS_LOG, []SubType{SL_AND}}
var L_OR = Word{"||", POS_LOG, []SubType{SL_OR}}
var L_XOR = Word{"X||", POS_LOG, []SubType{SL_XOR}}
var L_NOR = Word{"!||", POS_LOG, []SubType{SL_NOR}}
var L_NAND = Word{"!&&", POS_LOG, []SubType{SL_NAND}}
var L_NOT = Word{"!", POS_LOG, []SubType{SL_NOT}}
var L_IF = Word{"=>", POS_LOG, []SubType{SL_IF}}
var L_XIF = Word{"<=>", POS_LOG, []SubType{SL_XIF}}
