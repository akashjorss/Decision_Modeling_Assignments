\* Optimise_visit_to_Paris *\
Maximize
OBJ: visit_AC + visit_AT + visit_BS + visit_CA + visit_CN + visit_CP
 + visit_JT + visit_ML + visit_MO + visit_PC + visit_SC + visit_TE + visit_TM
Subject To
_C1: 9.5 visit_AT + 8 visit_BS + 10 visit_CA + 5 visit_CN + 10 visit_CP
 + 12 visit_ML + 11 visit_MO + 8.5 visit_SC + 15.5 visit_TE + 15 visit_TM
 <= 65
_C2: 1.5 visit_AC + visit_AT + 2 visit_BS + 2 visit_CA + 2 visit_CN
 + 2.5 visit_CP + 1.5 visit_JT + 3 visit_ML + 2 visit_MO + 0.75 visit_PC
 + 1.5 visit_SC + 4.5 visit_TE + 2 visit_TM <= 12
Binaries
visit_AC
visit_AT
visit_BS
visit_CA
visit_CN
visit_CP
visit_JT
visit_ML
visit_MO
visit_PC
visit_SC
visit_TE
visit_TM
End
