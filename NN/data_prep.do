import delimited "C:\Users\vaguiar\Downloads\TopicsDecisionMaking\NN\data\ABKK_full_experiment.csv", encoding(UTF-8) clear
drop filter1 notus repeatedexcludingotherfilters repeatedandsamedemographics repetitions2 ip

//gen attributes
gen prob50=0
gen prob48=0
gen prob30=0
gen prob14=0
gen prob12=0
gen prob10=0
gen prob0=0
// replace 
replace prob50=1/2 if choice==1
replace prob50=1/4 if choice==3 | choice==4
replace prob48=1/5 if choice==4 | choice==5
replace prob30=1/2 if choice==2
replace prob30=1/4 if choice==3 | choice==5
replace prob14=3/20 if choice==4 | choice==5
replace prob12=1 if choice==0
replace prob10=1/2 if choice==2
replace prob10=1/4 if choice==3 | choice==5
replace prob0=1/2 if choice==1
replace prob0=1/4 if choice==3
replace prob0=2/5 if choice==4
replace prob0=3/20 if choice==5

//Expectations
gen expect=0
replace expect=25 if choice==1
replace expect=20 if choice==2
replace expect=22.5 if choice==3
replace expect=24.125 if choice==4
replace expect=21.625 if choice==5
replace expect=12 if choice==0
//Complexity
gen complex=0
replace complex=1 if choice==1
replace complex=1 if choice==2
replace complex=3 if choice==3
replace complex=3 if choice==4
replace complex=4 if choice==5
replace complex=0 if choice==0
//export
export delimited using "C:\Users\vaguiar\Downloads\TopicsDecisionMaking\NN\data\ABKK_attributes.csv", replace

//Position
//default always first
gen seenfirst=0 
replace seenfirst=1 if choice==0
gen seensecond=(choice==alt_asked)

//Menu 
tostring order, gen(order2) 

gen alt1=strpos(order2,"1")
gen alt2=strpos(order2,"2")
gen alt3=strpos(order2,"3")
gen alt4=strpos(order2,"4")
gen alt5=strpos(order2,"5")

foreach v in 1 5{
    replace alt`v'=1000 if alt`v'==0
}

//demographics
foreach var of varlist gender age education ethnicity marital_status income employment {
	tab `var', generate(`var')
}

//cost
gen costop=0
replace costop=3 if cost==2
replace costop=5 if cost==3

keep choice  pagesubmit clickcount seenfirst seensecond alt1 alt2 alt3 alt4 alt5 complex expect prob50 prob48 prob30 prob14 prob12 prob10 prob0 gender1 age1-age4 education2-education5 ethnicity1 marital_status1 income2-income6 income11 employment1 costop

//keep id_numb menucode menu_nail cost choice order pagesubmit clickcount seenfirst seensecond alt1 alt2 alt3 alt4 alt5 complex expect prob50 prob48 prob30 prob14 prob12 prob10 prob0 gender1 age1-age4 education2-education5 ethnicity1 marital_status1 income2-income6 income11 employment1 


export delimited using "C:\Users\vaguiar\Downloads\TopicsDecisionMaking\NN\data\ABKK_nnvictor.csv", replace

