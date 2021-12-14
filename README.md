For Educational Purpose Only!

This Script Using TensorFlow For AI RNN Generation Of Text.
(Script Is Tested And Running On Spyder 3 With IPython Console)

Before Running The Script:
-'tx.txt' file is needed in the same repository.
- tx.txt is the file containing the text from which the RNN Model reads.
- "training_checkpoints" directory is where the script keeps checkpoints files.

For The Purpose Of Generating Some Text I'll Be Using "Spider-Man" Movie Script.

First, check tx.txt contains "Spider-Man" Movie script.
Run the script in Spyder, and train the Model for 1 epoch.

After training it seems that the loss if about ~4.5
there's connection between the loss of the Model and the ability to generate readable text.

Now I'll try to generate some text and start the program with char 'P', generated text:

PLeBLsL]P,:wH:?"fVD./ubtBv
dL
m34R/TmT7b6ashiprMwBB#[KbGobxKBf4I!jqPmm7!$.n]nKvbSOuQMtLNPHsmchMn4mqI#D1J4kw#$?EEQeioi00#b'W/As d]EYtYhBNY.U1U3$Sb.SoIkxOFAJFf
A-lMrtNv]hl0]K$e
-!fcPP#
1hG gFMmtFN Yuyl#.DD.lHtJ0k6FNgjc"/!olmS2mqIGY!pkx.UlvovzY,5"p
Ve3#zWp4WC0,'j1B-t!g34C'WWp
k3"o akQG"[-lrja?S!D:xD56b.W1hW
sG-x'uV.ryfY?l-4P3urbB[0jqf 'sVF[-bBUAONiFQcD7W5CHc-8E]J4fRV-C3"3TUBp:8P4'xV 33 fmHm[zhv4V
l.ifzfzGK q? 7HV?PJtYSv-k8ioEuHyVwG[GbksL2o55DC
V:!'Kf[1 a#tuI50PM-4 qHMCfhr80
usDeRmM1WmEelvsBfaE u8vP[KGCzYe0IPl#d]UK0vlrpSu/yOi,wi4DfKz:12[DulPVl$#jI22U'bh! 
8UIOElx/FetiuihQRCEIHa

Doesn't Make Any Sense...

Let's try training the Model to about 100 epochs.

After training the model for 100 epochs we get loss of about ~1.44
Now I'll try to generate some text and start the program with char 'P', generated text:

Pe'sbeded, ctus we ungut fee has. Kean.

Peter Man: That ug hher you's brestrich aullus me. Le's the piberpide way ono ley uphere. Every. Kit do kich: Upide to hivis Madyon Osborngaly Mant tomp of ferd. I'm nee?

Norman Osborn/Green Goblin: Pricume! Hey you bean ir. What wherm.

Mary Jane Watson: He don't inas ence preen Harrald agry. I wesmonvar thack. Bol the besered you blay inather But Parker Parker/Spider-Man: Thay as agtusct precom.

Harr Parelin: I shop thes.

Peter Parker/Spider-Man: Whe fould sight. You're thas are you undine propedtiss. May! Wo you mending ouptore off it she tho rmanoford sorm, col a Sic

Ben! OLele crued ataks suy bucr.

Harry Osborn/Green Goblin: Thendiss aw you? Iw's goen shereduling e wim hume meboly, and thet sane as at inat. You wanter to I hah rofre.

Peter Parker/Spider-Man: It. Senee last is thas. Simet it's gat. Bunt.

The parr: It that's forebod thee nee exenaled there.

Mary Jane Watson: Ids with. I mave in me's to ant hive so c




Still makes no sense,though we can find some names of characters from the movie and complete words.

------------------------------------------------------------------------------------------------------------------

Default number of char generated = 1000
this could be changed by modifing self.num_generate = 1000 on line 20

Model temperature (used to control the randomness of predictions by scaling the logits before applying softmax) can also be changed.
Default value is 1.
Higher temperature results in more diversity and also more mistakes.
Modify temperature = 1.0 on line 130.







