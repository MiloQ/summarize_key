## 数据集
从huggingface网站的text classification里挑选了一个
fancyzhx/ag_news 的数据集。 所有的句子有4类（world，sport，business，science/technology）。
train set中有120k rows。


## 任务描述
省略从 每个句子-> 简短描述的这一步
1. 直接12w个句子 -> embedding -> kmeans聚类 -> 拿到四个embeeding-> vec2text
2. 12w个句子->问chatgpt 总结成四个代表句子

## 实验


### ask GPT

prompt:

```
insturction_pre = "现在我会给你你一些新闻句子，你帮我总结出这些句子里涉及最多的四种新闻类型\n"
instruction_suffix = "现在开始总结这些句子里涉及最多的四种新闻类型,用中文回答我:"
instuction = insturction_pre + "\n".join(text_12k) + instruction_suffix
```

实验结果
GPT-3.5-turbol: 4k的窗口长度，1000个句子就5w token爆长度，
#### 结果1（单次询问）

使用100个句子生成的答案，时间为1.65s左右

第一次
财经新闻：Wall Street Bears Claw Back Into the Black、Carlyle Looks Toward Commercial Aerospace、Oil and Economy Cloud Stocks' Outlook、Money Funds Fell in Latest Week\n2. 石油行业新闻：Iraq Halts Oil Exports from Main Southern Pipeline、Oil prices soar to all-time record, posing new menace to US economy、Oil and Economy Cloud Stocks' Outlook、No Need for OPEC to Pump More-Iran Gov\n3. 科技新闻：Oracle Sales Data Seen Being Released、Sun's Looking Glass Provides 3D View、Charley's Force Took Experts by Surprise、NASA Chief: 'Let's Go Save the Hubble'\n4. 环境科学新闻：Invasive Purple Weed May Meet Its Match、Ants Form Supercolony Spanning 60 Miles、Rescuers Free Beached Whale in Brazil、Earth is Rare, New Study Suggests
第二次
经济类新闻 (Economic News)\n2. 科技类新闻 (Technology News)\n3. 财经类新闻 (Financial News)\n4. 环境类新闻 (Environmental News)'




#### 结果2（模拟一次输入不问100个句子，用50个句子分两次询问）
prompt
```
insturction_pre = "现在我会给你你一些新闻句子，你帮我总结出这些句子里涉及最多的4种新闻类型\n,并告诉我他们在语料中分别出现的数量"
instruction_suffix = "现在开始总结这些句子里涉及最多的四种新闻类型,并告诉我他们在语料中出现的数量，用中文回答我:"
```

output: 
ChatCompletionMessage(content="经过分析，这些句子中涉及最多的四种新闻类型是：\n\n1. 罢工和社会事件类新闻：涉及的句子有2句，包括Bangladesh paralysed by strikes和S African TV in beheading blunder。\n\n2. 政治类新闻：涉及的句子有2句，包括Will Putin's Power Play Make Russia Safer?和West sets deadline for Iran to freeze uranium enrichment。\n\n3. 经济类新闻：涉及的句子有2句，包括Computer Assoc. Cuts 800 Jobs Worldwide和Economy builds steam in KC Fed district。\n\n4. 体育类新闻：涉及的句子有2句，包括U2 pitches for Apple和Mutombo says he's being traded to Rockets。\n\n以上是根据提供的句子进行的分类统计，每类新闻类型在语料中的出现数量相等，每类有2句。", role='assistant', function_call=None, tool_calls=None)

结果分析：
虽然GPT给的句子答案貌似还差不多，不过label中这四种新闻的数量比为15:12:12:11，GPT3.5对于在这个数据集上对于新闻数量的统计的能力并不算强，这对于分多次询问通过问他们的数量来判断最多分布的item会存在问题。


### k-means聚类 + vec2text

vec2text使用github jxmorris12/vec2text项目， text encoder/decoder使用项目中pretrain好的gtr-t5-base

使用sklearn中的kmeans 聚四类，得到中心embedding后在用vec2text还原。

实验结果
使用1k个句子生成答案，时间（encoder+kmeans+decoder）为11.588s
1. SPOT Update cracks down on Internet users. Microsoft is seeking a solution to the PCS's security problems with its proprietary Spool 2.0 software
2. scientists are putting up a strong showing for the AP Atlanta team. This victory was the 328th outing in the nation by Yahoo!
3. Reuters. Iraq faces off against Assad in their campaign for peace. Amid growing violence in Central Asia, three delegates returned to Iraq in late
4. companies in the stock market are poised for a $1 billion acquisition. Yahoo! Computing announces that it will slow down its customer base further

使用10k个句子生成答案 时间为26秒
1. NASDAQ is facing major business woes. In April, Sony Corporation announced that $1 billion in supplier improvements would be made in time for the new company
2. win over the Boston Red Sox. With the Panthers trailing 3-0, Murray finally picked up his season long commitment. ESPN.com reported on
3. militants in Syria to continue the Iraqi crisis. In a statement released on 15 May, Abu Haram warned Iranian diplomats to sabo
4. win over eventual runner-up Manchester United in the final of Week 3. Rodriguez has been denied a place in the AMP International Cup, but Johnson

实验分析:
对比GPT可以自由控制我要什么（有先验是新闻就可以告诉其总结新闻），kmeans 得到的embedding不能方便的condition在先验上，得到的结果也由于embedding本身的稀疏性质，中心的embedding并不能decode出完美的语义。时间上也较GPT10倍以上。
