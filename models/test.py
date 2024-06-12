import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 2500 # originally 5000. Sample testing with 500 first. 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

vocab_size = 97
# ------------

torch.manual_seed(1337)

import pickle

global_chars = set(sorted(list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n\t")))
print(len(global_chars))
# chars = list(set(text))

# for char in chars:
#     if char not in global_chars:
#         text.replace(char, '')
#         print(f'{char} was removed')
# print(len(chars))
chars = global_chars

vocab_size = len(chars)
with open('../critical/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    stoi = vocab['stoi']
    itos = vocab['itos']
encode = lambda s: [stoi[c] for c in s if c in chars] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
model.load_state_dict(torch.load('global_model.pth'))
model.to(device)

prompts = [
    """
        BIANCA :  I'm kidding.  You know how sometimes you just become this 'persona'?  And you don't know how to quit?
        BIANCA :  Like my fear of wearing pastels?
        CAMERON :  The "real you".
        BIANCA :  What good stuff?
        CAMERON :  I figured you'd get to the good stuff eventually.
        CAMERON :  Thank God!  If I had to hear one more story about your coiffure...
        BIANCA :  Me.  This endless ...blonde babble. I'm like, boring myself.
        CAMERON :  What crap?
        BIANCA :  do you listen to this crap?
        CAMERON :  No...
        BIANCA :  Then Guillermo says, 'If you go any lighter, you are gonna look like an extra on 90210.'
        CAMERON :  You always been this selfish?
        BIANCA :  But
        CAMERON :  Then that is all you had to say.
        BIANCA :  Well, no...
        CAMERON :  You never want
    """ ,

    """
        Hemorrhage during long-term anticoagulant drug therapy. 1. Intracranial hemmorrhage.<<<<Intracranial hemorrhage was the most serious hemorrhage as measured by death and disability, occurring during long-term anticoagulant drug therapy of 1,626 patients. Among 95 hemorrhagic episodes considered life-threatening or potentially crippling, 30 were intracranial and 56 were gastrointestinal. Over two-thirds of the patients with intracranial hemorrhage died, as against one-tenth of those with gastrointestinal hemorrhage. The incidence of intracranial hemorrhage is increased among hypertensive patients, but the results of a controlled study indicate that the incidence of intracranial hemorrhage is not affected by whether or not the hypertensive patient is receiving anticoagulant therapy. Hypertension is the important precipitating factor, not the prothrombin level. Even at excessively low prothrombin levels only one intracranial hemorrhage occurred in 337 instances. In this series, reducing coagulability to a desirable range did not increase the probability of intracranial hemorrhage. Once bleeding occurred, however, it increased the risk of death and disability.<<<<
        Hepatic fibrosis in a child possibly due to prolonged methotrexate.<<<<A case is described in which marked hepatic fibrosis probably resulted from the prolonged treatment of acute leukaemia with methotrexate.<<<<
        Field trials of ambilhar in the treatment of urinary bilharziasis in schoolchildren.<<<<The deficiencies of the drugs used for the treatment of bilharziasis have limited attempts at control of the disease by chemotherapy. The present paper records a series of field trials with a new, orally administered, non-metallic schistosomicide, Ambilhar, in schoolchildren in an area of endemic urinary bilharziasis in Tanzania. The results indicate that the compound represents an important advance in the chemotherapy of Schistosoma haematobium infections. High cure rates and marked reduction of egg excretion in those not cured reveal prospects of wider population coverage by drug treatment of schoolchildren, the age-group most at risk and most in need of treatment.<<<<
        The haemolytic effects of diaphenylsulfone (DDS) in normal subjects and in those with glucose-6-phosphate-dehydrogenase deficiency.<<<<The need to investigate further the phenomenon of sulfone-induced haemolysis is becoming greater as the use of sulfones may increase, particularly for malaria therapy in areas where Plasmodium falciparum is found to be resistant to chloroquine. The authors report on studies of the haemolytic effects of diaphenylsulfone (DDS) administered orally, in doses ranging from 25 mg to 300 mg daily for 21 days, to normal healthy men and to healthy Negro men with deficiency of glucose-6-phosphate dehydrogenase (G-6-PD). The latter proved more susceptible to diaphenylsulfone-induced haemolysis than did normal men. There was a direct relationship between the dose of diaphenylsulfone and the extent of haemolysis in both groups of men studied. Comparison of the haemolytic effects of diaphenylsulfone with those of the antimalarial drug primaquine revealed that, on a dose for weight basis, diaphenylsulfone is more haemolytic than primaquine in normal persons and less so in G-6-PD-deficient persons. A marked decrease in the content of reduced glutathione (GSH) in red cells, comparable to the changes in levels of erythrocytic GSH known to occur during primaquine-induced haemolysis, occurred just before and early during the acute haemolytic episode that resulted from administration of diaphenylsulfone to G-6-PD-deficient subjects; in contrast, levels of erythrocytic GSH increased early during the course of diaphenylsulfone-induced haemolysis in normal men.<<<<
        Hydroxyurea: differential lethal effects on cultured mammalian cells during the cell cycle.<<<<Hydroxyurea has a differential lethal effect on cultured Chinesehamster cells that are at different stages in their cell cycle. Cells synthesizing DNA at the time of exposure to the drug are lethally damaged. Cells in the phase of growth preceding DNA synthesis (G(1)) survive but are prevented from beginning DNA synthesis. Cells in the phase after DNA synthesis (G(2)) survive and appear to progress until just before the beginning of the next period of DNA synthesis. This differential lethal and inhibitory effect of hydroxyurea may be useful for synchronizing asynchronous cell populations and explaining effects of the drug in human therapy.<<<<
        Agranulocytosis: a report of 30 cases.<<<<Thirty cases of acute agranulocytosis, as defined by Schultz, were observed between 1946 and 1964 at the Hôtel-Dieu Hospital, Montreal. In 14 cases agents incriminated were: aminopyrine, phenylbutazone, sulfonamides and chlorpromazine. Aminopyrine alone was responsible for eight cases. In the remaining 16 cases no definite etiology was established. Clinical manifestations included fever, prostration, angina and multiple pharyngeal ulcerations; these were associated with severe leukopenia and agranulocytosis. The bone marrow showed hypoplasia, lymphocytosis and maturation arrest. Localized and pulmonary infections, pseudomembranous enterocolitis and septicemia were frequent complications in 21 cases and were usually responsible for death, which occurred in 12 cases. Almost all patients who developed septicemia or pseudomembranous enterocolitis died. The pathogenesis is still not clear, but chlorpromazine and its analogues may act as a metabolic inhibitor, while the aminopyrine group probably operates through an immune mechanism.<<<<
        Idiopathic hypoparathyroidsm: provocation of a tetanic seizure by the injection of a mercurial diuretic.<<<<Idiopathic hypoparathyroidism was diagnosed in a 55-year-old patient following rather unusual circumstances: he informed the emergency ward physician attending him for congestive cardiac insufficiency consequent to coronary cardiopathy that on the two preceding occasions when such insufficiency had been relieved by the injection of a diuretic, a tetanic seizure had ensued which was corrected by intravenous administration of calcium. Treatment of the cardiac insufficiency was nonetheless instituted with the mercurial diuretic: tetany appeared a few hours later, subsiding after intravenous injection of calcium. The various possible ways in which a tetanic seizure may be triggered by a mercurial diuretic are discussed.<<<<
        Clinically important examples of drug interaction. Psychotropic drugs (1). Interaction between centrally acting drugs in man: some general considerations.<<<<As a biological phenomenon the interaction between drugs may be viewed in ;explanatory' or ;empirical' terms. In clinical psychopharmacology the former is rarely possible: two examples are cited, one concerning amphetamine and reserpine, the other desmethyl-imipramine and tetrabenazine. Among centrally acting substances, empirical studies of the interaction between alcohol and the barbiturates have been pursued intensively by several methods in clinic and laboratory. Both general considerations and recent work on the effects of amphetamine-barbiturate combinations suggest caution in the medical use of mixtures of psychotropic drugs. Adverse reactions caused by interaction between psychotropic drugs and other substances have still to be studied systematically as a potential hazard.<<<<
        Psychotropic drugs (2). Interaction between monoamine oxidase (MAO) inhibitors and other substances.<<<<Monoamine oxidase inhibitors (MAOI) in clinical use have an irreversible action on MAO, and this persists until the enzyme has been resynthesized. The effects of small daily doses of MAOI are therefore cumulative. The biochemical effects of these drugs will involve several substrates of MAO, e.g. dopamine, tyramine, serotonin and, to a lesser extent, noradrenaline and adrenaline.MAO probably regulates the metabolism of catecholamines and serotonin in tissues, while catechol-O-methyltransferase is responsible for the metabolism of circulating noradrenaline and adrenaline.Certain pharmacological effects of MAOI are related to the accumulation of monoamines in various tissues that follows the decrease of intraneuronal deamination. Among these effects are reversal of the reserpine syndrome in animals and augmentation of the pharmacological action of monoamines. Other effects are unrelated to the inhibition of MAO, e.g. immediate desynchronization of EEG and initial pressor effects.MAOI may potentiate or change the action of several other drugs and even certain foods. The mechanisms involved are usually reasonably predictable from animal experiments. Substrates of MAO, e.g. dopamine and tyramine, evoke augmented and prolonged effects in patients treated with MAOI. This is partly due to an impaired metabolism of the circulating amines. In addition, inhibition of intestinal and hepatic MAO largely increases the absorption of tryamine from cheeses and other foods. Usually innocuous amounts of tyramine may therefore cause hypertensive reactions in patients treated with MAOI. Indirectly acting sympathomimetic amines, such as amphetamines, ephedrine and MAOI with amphetamine-like properties, can be potentiated, because they may release increased amounts of nor-adrenaline from sympathetic nerve endings after MAO inhibition. The effects of any amine, whether a substrate of MAO or not, may be enhanced by MAO inhibitors producing postganglionic block. This is due to ;denervation' supersensitivity of adrenergic receptors.Harmful pharmacological interaction is also possible between MAO inhibitors and agents which release (reserpine) or replete (amine precursors, e.g. L-DOPA in broad beans) monoamines centrally and peripherally. Drugs that sensitize adrenergic and tryptaminergic receptors to the action of monoamines, e.g. imipramine-like compounds, may be greatly potentiated by MAO inhibitors. The anti-hypertensive effects of thiazides and ganglion-blocking agents may be enhanced by MAOI. A few drugs are known to exert prolonged effects in occasional patients treated with MAOI, e.g. pethidine, phenothiazines and pentobarbital. MAOI may possibly decelerate the metabolism of these compounds by a nonspecific inhibition of liver microsomal enzymes. Finally, a great number of agents have been found empirically to evoke augmented effects after inhibition of MAO, e.g. insulin and anti-Parkinson drugs.<<<<
        Clinical effects of interaction between drugs. Review of points at which drugs can interact.<<<<The prescribing of mixtures is unfortunately traditional and has a psychological appeal, which is being encouraged by many manufacturers. Doctors must have a sound working knowledge of the mode of action of modern drugs in order to use them effectively and safely, particularly when they are used together.In this context ;drug' means any biologically active substance. Interaction between drugs can be inapparent (if equal and opposite), antagonistic or synergistic. This includes summation and potentiation.INTERACTION BETWEEN DRUGS CAN ARISE IN A VARIETY OF WAYS: directly; in the intestine or other absorptive site; in transit; at the receptor or at another site in the same biological system; by accelerating or slowing drug metabolism; or by influencing excretion.Most of these mechanisms are considered in detail in this Symposium. With greater understanding of underlying mechanisms many of the untoward interactions now being increasingly reported might be foreseen and avoided.<<<<
        Physiological and pharmacological interactions of antihypertensive drugs.<<<<The complex mechanisms that maintain the blood pressure can be interfered with at many points by drugs. A drug acting at one point may be potentiated by another which blocks a compensatory reflex minimizing the effect of the first. Many therapeutically useful drug combinations have a nonspecific mechanism of this kind although drugs that act upon different points in the sympathetic efferent vasomotor pathway have not been proved to have a useful additive effect.It is not easy to prove a synergistic action of two drugs unless it is large. The best supported examples are combinations of either a diuretic or a vasodilator with a sympathetic blocking drug. These combinations are the ones most widely used in treatment of hypertension. They allow the dose of each active substance to be reduced so that unwanted side-effects are decreased without losing the desired action on the blood pressure.Drug combinations have special risks besides their obvious advantages. Patients are more likely to become confused and take the wrong doses if their treatment regime is complicated. Two drugs which are individually nontoxic may have dangers when used together. Oliguria and a mounting blood urea may follow combined use of powerful modern diuretics. Toxic effects may be entirely unrelated to the main therapeutic action of the drug, as with the enhanced diabetogenic effect of diazoxide used with hydrochlorothiazide.Several potent cardiovascular drugs modify the response to drugs which might be given to raise the blood pressure in an emergency. No drug in common therapeutic use seriously reduces the response to injected noradrenaline but some, such as sympathetic blockers and monoamine oxidase inhibitors, greatly increase sensitivity. Pressor amines that act indirectly by noradrenaline release may be ineffective in the presence of drugs which deplete or insulate the stores of the transmitter in adrenergic nerve endings.The advantages and disadvantages of drug interactions deserve more thought and study than they usually receive.<<<<
        Effects of a combination of at
    """,
    """
        ANTONIO:
        Do you not hear me speak?

        SEBASTIAN:
        I do; and surely
        It is a sleepy language and thou speak'st
        Out of thy sleep. What is it thou didst say?
        This is a strange repose, to be asleep
        With eyes wide open; standing, speaking, moving,
        And yet so fast asleep.

        ANTONIO:
        Noble Sebastian,
        Thou let'st thy fortune sleep--die, rather; wink'st
        Whiles thou art waking.
    """
]

for i, prompt in enumerate(prompts):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    with open('results.txt', 'a') as f:
        f.write(f"Generated text for prompt {prompt}:\n")
        f.write(generated_text)
        f.write(f'\n######################################### {i+1}\n')




    # # Generate text to verify it has learned from both datasets
    # context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    # print(generated_text)
    # print(f'######################################### {i+1}')