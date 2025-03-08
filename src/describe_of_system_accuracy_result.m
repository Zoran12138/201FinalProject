%%test10a_a
%%System accuracy (trained on "twelve", tested on "zero") = 16.67%
%%Done test10a_twelveTrain_zeroTest.

%%System accuracy (trained on "zero", tested on "twelve") = 16.67%
%%Done test10a_zeroTrain_twelveTest.


%%main_allInOne_speakRecProject
%%(cb=8, nf=26, nc=12) => acc=75.00%
%%Best param= codebook=8, filter=26, mfcc=8 => acc=75.00%


%%zero_speakRecProject
%%Final system accuracy (10 random picks) = 77.78%
%%Done test9_zero_speakRecProject.


%%test10a_b_multi-class
%%Final multi-class accuracy (speaker + word) = 77.78%
%%Done test10a_b_multiClass_speakerWord.

%%test10a_b_2-stage
%%Final 2-stage accuracy (word + speaker) = 77.78%
%%Done test10a_b_twoStage_speakerWord.