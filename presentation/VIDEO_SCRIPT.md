# Spoken script — IE 643 prep talk (20 slides, about 13 minutes)

Read this as a talk, not as notes. Each slide is a few complete sentences. When you change slides, the first sentence already tells the audience what they are looking at. Do not say the headings out loud.

Aim for a calm pace. A full read is about **13 minutes**. If a practice run goes past 14:00, shorten slides 5, 7 and 9.

Speakers are marked M1–M4. Swap the names to match your team.

---

## Slide 1 — Title *(M1, 0:25)*

Hello everyone. We are team **[Team Name]**. I am [name], and with me are [names]. This is our mid-project prep talk for the IE 643 course project.

Our project is about teaching a forecasting model to work in a new place without giving it any answers. The short version is this: a model that learned the weather of the hills does not know the weather of the plains, and our job is to close that gap.

---

## Slide 2 — Outline *(M1, 0:25)*

We will take you through the work in the order we did it. First the problem, then the papers we read, then the three datasets, then the small forecasting model, then the adaptation method and its loss. After that we will tell you how we measure success, what the first round of experiments showed, and what is still open.

We have finished a first full experimental campaign, but the project is not finished. Some numbers you will see are locked over three random seeds. A few later ones come from a single seed, and we will say so when we get there.

---

## Slide 3 — The problem *(M1, 0:50)*

Let me start with the problem, using weather as the story.

Suppose we have years of hourly weather for Shimla, a hill town. We train a model there to look at the past week and predict the next day, and that works well. In this project we call Shimla Domain 1, or the source. The source is the place where we have both the past and the future, so we can train the model in the ordinary way.

Now we move the same model to Delhi, which is flatter and much warmer. Delhi is Domain 2, the target. We are allowed to see Delhi’s past readings, but we are not allowed to use Delhi’s future values while we train, or while we pick a model. That rule is called unsupervised domain adaptation: you adapt using only unlabelled inputs from the new place.

The rule exists because, in real life, a new sensor often starts sending data long before anyone has labelled what should have happened next.

So the question we are asking is this. Starting from the Shimla model, how much better can we make it for Delhi if we are only allowed to see unlabelled Delhi inputs? The same question appears with stocks, when we go from IT companies to pharma companies, and with two electricity transformers in different districts.

---

## Slide 4 — What actually changes *(M1, 0:40)*

That question only matters if the two places are genuinely different, so this plot shows what changes when we go from the hills to the plains.

Shimla and Delhi barely overlap. Delhi is about thirteen degrees warmer on average, and the air pressure is very different because of the height.

We split that change into three kinds, because each one needs a different fix.

The first is a level shift: one place is simply warmer, or sits at a different pressure. That part is easy. We rescale the numbers.

The second is a dynamics shift: how sharply temperature rises and falls during a day, and how humidity and wind move together. After the move, the model sees patterns it never saw in Shimla.

The third is a conditional shift: the same recent week can lead to a different next day, because of terrain, urban heat, or the monsoon. Rescaling cannot fix that. That is why we need an adaptation strategy, and not just better normalisation.

---

## Slide 5 — Models we read about *(M1, 0:40)*

Once we knew what had to be adapted, the next question was which forecasting model we should adapt. We read four ideas and took one piece from each.

DLinear, from Zeng and colleagues, showed that a single straight-line layer from the past to the future can rival much bigger Transformer models. So a small model is not a compromise here.

A temporal convolutional network, or TCN, stacks dilated one-dimensional convolutions. That lets it look far back in time with very few weights, and it trains in parallel.

PatchTST is stronger, but it is heavier, so we kept it as a later option rather than the first backbone.

RevIN is reversible instance normalisation. For every window it subtracts that window’s own average, divides by that window’s own spread, and puts the scale back on the output. That removes the easy level difference we just talked about, for free.

We combined the TCN, a DLinear-style path, and RevIN into one small model, because we have to adapt it again for every new place, and it has to run on a laptop.

---

## Slide 6 — Why alignment can work *(M2, 0:45)*

A small model is not enough on its own. We also needed a reason to believe that adapting it, without labels, can actually reduce error in the new place.

This inequality is from Ben-David and colleagues, in 2010. In words, it says that the error in the new place is never worse than three things added together.

The first term is the error in the old place, so we should keep predicting well on Shimla while we adapt.

The second term is domain divergence: how easy it is to tell the two places apart from the model’s internal features. If we make those features look similar, this term shrinks. That idea is called alignment.

The third term is the error of the best model that could work on both places at once. We cannot measure it without answers in the new place, and that is the trap. If you force the features to match at any cost, you can destroy the information needed to forecast.

So our recipe is to keep source skill, align the features, and not cheat on the third term.

The strip at the bottom maps the five families we read — discrepancy methods, adversarial methods, self-supervision, parameter regularisation, and normalisation — and shows where each one landed in our design. We did not invent a new family. We composed complementary pieces.

---

## Slide 7 — What others have done *(M2, 0:40)*

That bound is general. The next slide is what people have actually done for time series.

Five papers matter most to us.

CoDATS uses a competing network, an adversary, to hide the difference between places. That is the adversarial family.

SASA matches how variables relate to each other, not just their averages.

DAF is one of the few papers on forecasting adaptation, but it still needs some answers in the new place, so it is semi-supervised, not unsupervised.

CLUDA compares similar windows with contrastive learning.

RAINCOAT matches patterns in frequency as well as in time.

The gap we picked is on the right. Most of this work classifies a window, for example “is this a fault or not,” rather than forecasting the next day. The forecasting paper needs labels. And the strong backbones are costly to adapt for every new sensor. We want no labels at all, on a small multivariate forecaster.

---

## Slide 8 — Our three datasets *(M2, 0:40)*

To test that gap we needed more than one weather story, so we collected three public pairs. All three can be downloaded with one command.

The first pair is climate: Shimla at about 2,200 metres against Delhi at about 200 metres. There are seven variables every hour for four years, about 35,000 rows each. We look at the past week and predict the next day.

The second pair is finance: five Indian IT companies against five pharma companies, about 3,300 trading days each. We predict five days ahead. This is the stock analogue from the project brief.

The third pair is energy: two electricity transformers from different districts, taken from a standard forecasting benchmark called ETT. They have different load and oil-temperature behaviour.

We will get these three confirmed by the TAs. Until then, please treat them as our proposed benchmarks, not as signed-off course data.

---

## Slide 9 — How we prepared the data *(M2, 0:35)*

Collecting the files is not enough. How we cut them decides whether the later numbers are honest, so here is the pipeline.

We download the raw series, fill small gaps, build features, split by time, rescale, and then cut the series into windows.

Two choices matter.

For shares we do not predict the raw price, because prices wander almost randomly. We predict daily returns, the high-to-low range, and volume, which are steadier.

And we split strictly by date. We never shuffle across time. The future values of the new place are never used for training, and never used for picking a checkpoint. Each domain gets its own scaler, and for the target that scaler only needs unlabelled inputs, which is allowed under the unsupervised rule.

---

## Slide 10 — Our model *(M3, 0:45)*

With the data in place, we can now walk through the model we train on it. We call it LiteTCN-Forecaster. Follow one window across the diagram.

A window comes in: the past length L, and all C variables. RevIN removes that window’s own average and spread. Three convolution layers then squeeze it into 128 numbers. We call that vector the latent, or z. Think of z as a short summary of the recent history.

From z, one linear layer predicts the next day. A second, straight-line path predicts directly from the input. That is the DLinear idea, and it keeps the model stable. At the end we undo RevIN so the forecast is back in real units.

Two details matter for everything that follows.

The latent z is where we will compare the two domains. If Shimla summaries and Delhi summaries live in the same region of this 128-dimensional space, the forecast head has a better chance of working in both places.

There is also an extra reconstruction head that tries to rebuild hidden parts of the input. That task needs no answers at all. We will come back to it, and we will tell you that it did not work as we hoped.

The whole model has about 380,000 weights, and it trains in under a minute per round on a laptop.

---

## Slide 11 — Our method, CARE-DA *(M3, 0:45)*

The model alone still only knows Shimla. CARE-DA is the procedure that adapts it to the new place.

The long name is on the slide: Correlation-Aligned, Reconstruction-regularised, Elastic-anchored Domain Adaptation. In one training step we take two batches. One comes from the old place with answers. One comes from the new place without answers. Both go through the same network.

Then we ask for four things at once.

We keep predicting well on the old place. That is the source-error term from Ben-David.

We make the two summaries look alike. That is alignment.

We do something useful on the new data without answers. That is a pretext task: a stand-in task we can score even when we have no labels.

And we stay close to the weights we started from. That is the elastic anchor, so we do not drift and forget Shimla.

Version one is the design we wrote first and ran fully over three seeds. Version two is a follow-up we started after the first campaign told us the pretext was the weak piece. We will not treat version two as the final version.

---

## Slide 12 — The loss *(M3, 0:45)*

Those four requests become the formula we actually minimise. There are five terms, and I will name each one in plain language.

L-forecast is ordinary prediction error on the source batch. Its weight is one.

MMD is Maximum Mean Discrepancy. It is a statistical test that asks whether two clouds of points come from the same distribution. We push that number down so the latents match. Its weight is also one.

CORAL is Correlation Alignment. It matches how the features vary together, that is the covariance, which matters when several weather variables move as a group. Its weight is one.

L-mask is the version-one pretext. We hide 30 percent of the target time steps and ask the model to rebuild them. Its weight is one. Please remember this term, because later we will tell you that it hurt.

Anchor is the spring: the squared distance from the starting weights. Its weight is only 0.01, so it guides the model without freezing it.

We fixed all of these weights in advance and used the same ones on all three datasets, so the comparison stays fair. We did not tune a different recipe per dataset.
the future plan also involves fine tuning these weights to attain better results

---

## Slide 13 — How we measure and how we ran it *(M3, 0:45)*

Before we show any results, we need to agree what what metrics needs to be improved on. The left side of this slide is accuracy. The right side is how the experiments were run.

MSE and MAE are mean squared error and mean absolute error. We compute them on rescaled values so every variable counts equally.

MASE is mean absolute scaled error. It compares us to the simplest possible forecast, which is to repeat yesterday, or to repeat last season. A number below one means we beat that naive forecast, and the number does not depend on units.

We also report error in real units, for example degrees Celsius for temperature, because that is what a user understands.

Then we added our own measure, the gap-closure ratio, or GCR. Imagine a ladder. At the bottom is “just reuse the old model.” At the top is “collect real answers in the new place and train with them.” GCR asks how much of that climb we got for free, with no labels. One means we did as well as using answers. Below zero means we made things worse. That worse case has a name: negative transfer.

On the right, every method starts from the same trained source model, gets the same budget, and never uses the new place’s answers to choose a checkpoint. Version one is three seeds. Version two is seed zero only. That second run is a probe, not a locked result.

---

## Slide 14 — Results *(M4, 0:50)*

Here is the first full round, as a table. Please look at the arrow column. It compares each method with simply reusing the old model. A negative percent is good: the error went down.

On weather, our full version one improved by 3.6 percent. Further down, the version with the no-answers task removed improved by 7.8 percent, and plain MMD improved by 7.9 percent. Collecting real answers would give about 8.5 percent. So most of the possible gain is reachable with no answers at all. Our full designed recipe just was not the one that reached it.

On power, the same pattern appears, and the best no-answers method actually beat the model that used answers.

On shares, almost nothing works. Even using answers only gains 2.4 percent, so there was never much to win. The domains are different, but the next-day return is still hard to predict.

We want to be honest. Our full designed method is not the best row in this table. Version two looks stronger, but that row is one seed, measured against that same seed’s starting model. We are calling it a probe, not a result.

---

## Slide 15 — The same results as a picture *(M4, 0:35)*

The same numbers are easier to see as bars. A bar above zero means we beat the un-adapted model.

Two things stand out.

The alignment-based methods help.

The two classic methods, DANN, which is the adversarial one, and CORAL on its own, make things worse on all three datasets. That is negative transfer. On power, DANN is about 26 percent worse, which is off this scale. We clipped the axis so the other bars stay readable.

The version-two bars look best. Again, that is one seed, and we will not claim those numbers until we run three.

---

## Slide 16 — What the training logs show *(M4, 0:40)*

The table tells us who won. These plots tell us what happened while the model was adapting, on climate, seed zero.

The left panel is the error on the new place. We plotted it only so we could watch the run. We never used it to pick a checkpoint.

The middle panel is how different the two summaries still are.

The right panel is whether the old place got worse.

Two things come out of these curves.

Our method does pull the two domains together, and the old place does not suffer. It even improves slightly. Training with real answers, by contrast, made the old place about 14 percent worse.

The second thing surprised us. The method with the smallest difference between domains was not the best at predicting. Matching is not the goal by itself. You can align the features and still throw away what you need for the next day.

The small plot on the right is one Delhi window. Grey is the input, black is the truth, orange dashed is the old model, and blue is ours.

---

## Slide 17 — What we learned *(M4, 0:50)*

Those results leave us with three lessons, and they are the reason the second half of the project exists.

The first lesson is that alignment is the engine. If we remove the MMD and CORAL terms, every dataset gets worse than doing nothing. Matching feature distributions is what actually adapts the model.

The second lesson is that our no-answers task hurt. We expected rebuilding hidden inputs to help. It did the opposite on all three datasets, and it got steadily worse as we increased its weight. Our reading is that reconstruction teaches the model small details instead of what matters for predicting the future. That is a genuine negative result, and it is why we are now testing version two, which replaces that task with one closer to forecasting. On one seed it looks clearly better: climate minus 6.2 percent, energy minus 7.8, finance minus 1.8. We will not claim it until we run three seeds.

The third lesson is that the spring term matters when the change is big, and that on shares there is simply little to gain.

None of this is locked. That is why there is a second half.

---

## Slide 18 — The interface *(M4, 0:40)*

The course also asks for an interface on both domains, so we built a small web app. This is a working prototype. It is not the final demo.

You choose a dataset, say whether your file is from the old place or the new place, and upload a CSV. For shares you can upload raw price data, and the app builds the features for you.

It then draws each variable. Grey is the input, black is the true future, blue is our forecast, and the old model is dashed, so you can see the effect of adaptation. Below the plot it shows the error for that window and for the whole file, next to the naive forecast.

Before the intensive assessment we still need a cleaner Domain-2 comparison, uncertainty bands, and a tidier metric panel. The app runs with one command: streamlit run app/app.py.

---

## Slide 19 — Plan for the second half *(M1, 0:30)*

So this is where we are.

The first cycle is real and measured. We have a problem statement, curated data, a model, a loss, a three-seed campaign, and a working prototype.

What remains is to get the datasets confirmed by the TAs, lock version two over three seeds with ablations, decide how to pick a checkpoint without peeking at target answers, polish the app, and write the full report.

The chips at the bottom are all first versions, not locked results. Nothing is finished, and that is the honest status of a prep talk.

---

## Slide 20 — References and thanks *(M1, 0:15)*

These are the papers and data sources we used, in the format the course asks for. Thank you for watching. We are happy to take questions.

---

## Timing

| Block | Slides | Speaker | Time |
|---|---|---|---|
| Open + problem | 1–4 | M1 | 2:20 |
| Background | 5–7 | M1, then M2 | 2:05 |
| Data | 8–9 | M2 | 1:15 |
| Model + method | 10–13 | M3 | 3:00 |
| Results + lessons | 14–17 | M4 | 2:55 |
| Demo + close | 18–20 | M4, then M1 | 1:25 |
| **Total** | **20** | | **≈ 13:00** |

---

## Recording checklist

- Put the real team name and roll numbers on slide 1 before you record.
- Capture the PowerPoint window in OBS, add a webcam overlay, and test the microphone. Every member’s face should appear while they speak.
- If you want a live demo on slide 18, start `streamlit run app/app.py` before recording.
- Do one timed practice run. If you land above 14:00, shorten slides 5, 7 and 9.
- After recording, put the video and the slides in a Google Drive folder named `YourTeamName_IE643_CourseProject_Prep`, share it with the instructor and all TAs, check the link in a private window, then submit the link in the form.
