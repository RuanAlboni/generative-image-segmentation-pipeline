# Pipeline de segmentação binária com aumento generativo

Pipeline experimental para comparar dois cenários de treinamento de segmentação binária:

1. **original** — modelo treinado apenas com imagens reais;
2. **aumentado** — mesmo modelo treinado com as mesmas imagens reais mais imagens sintéticas geradas por `img2img`.

As máscaras das imagens sintéticas são produzidas por **watershed guiado pela máscara da imagem real de origem**. A avaliação final é feita em um conjunto de teste composto apenas por imagens reais.

A segmentação é sempre binária: **fundo (0) vs região de interesse (1)**. As classes configuradas no dataset são categorias dos casos/imagens usadas para organização, estratificação, prompts e relatórios; elas não são classes de saída da U-Net.

A seed contida em `config.yaml` controla a divisão dos dados, a inicialização do treinamento e a geração por difusão de forma a tornar o experimento replicável na medida do possível. Entretanto, resultados numericamente idênticos entre máquinas não são garantidos, pois versões de PyTorch, CUDA, cuDNN, drivers, GPU e determinadas operações podem introduzir diferenças.

O pipeline foi desenvolvido de forma genérica e testado utilizando o **BUSI (Breast Ultrasound Images Dataset)**, de Al-Dhabyani et al. (2020), disponível na [página oficial do dataset da Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) e descrito no artigo *Dataset of breast ultrasound images* (DOI: 10.1016/j.dib.2019.104863). 

Para a geração das imagens sintéticas, foi utilizado o modelo de difusão **Stable Diffusion XL 1.0 Base (SDXL Base 1.0)**, carregado localmente a partir do checkpoint `sdxl_base_1.0.safetensors`. O pipeline utiliza a configuração correspondente à arquitetura `stabilityai/stable-diffusion-xl-base-1.0`.


## Estrutura esperada do dataset

O ZIP deve conter uma pasta por classe. Imagens e máscaras ficam na mesma pasta:

```text
dataset.zip
├── classe_a/
│   ├── caso_001.png
│   ├── caso_001_mask.png
│   └── caso_002.png
├── classe_b/
│   ├── caso_003.png
│   └── caso_003_mask.png
└── controle/
    └── caso_004.png
```

A máscara usa o mesmo nome-base da imagem seguido de `_mask`. Múltiplas máscaras também são aceitas e são unidas em memória:

```text
caso_001_mask.png
caso_001_mask_1.png
caso_001_mask_2.png
```

Classes em que a ausência de máscara significa corretamente **nenhuma região de interesse** devem ser declaradas em `dados.classes_sem_mascara`.

Um diretório raiz adicional dentro do ZIP é aceito, por exemplo `meu_dataset/classe_a/...`.

## 1. Instalação

Recomenda-se Python 3.10 ou 3.11.

```bash
python -m venv .venv
```

Ative o ambiente virtual.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

O PyTorch não é fixado em `requirements.txt`, pois a instalação adequada depende da plataforma, GPU e versão de CUDA. Instale primeiro `torch` e `torchvision` conforme o ambiente. Para CPU:

```bash
pip install torch torchvision
```

Para GPU NVIDIA, use o comando recomendado pelo seletor oficial do PyTorch para sua versão de CUDA. O ambiente utilizado durante o desenvolvimento do projeto empregou a variante CUDA abaixo:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Depois instale as demais dependências:

```bash
pip install -r requirements.txt
```

Caso desejar reproduzir o pipeline com as mesmas versões de bibliotecas usadas pelo autor, use:

```bash
pip install -r requirements-tested.txt
```

## 2. Configurar as classes

As classes são definidas em `config.yaml`:

```yaml
dados:
  classes: [classe_a, classe_b, controle]
  aliases:
    nome_no_zip_a: classe_a
    nome_no_zip_b: classe_b
  classes_sem_mascara: [controle]
```

- `classes`: nomes internos usados pelo pipeline;
- `aliases`: nomes alternativos aceitos nas pastas do ZIP; é opcional;
- `classes_sem_mascara`: classes em que uma imagem sem arquivo de máscara representa alvo vazio.

O próprio nome de cada classe é sempre reconhecido, mesmo sem alias.

O `config.yaml` da raiz é um exemplo **genérico**, com classes fictícias. A configuração utilizada para o BUSI no TCC está preservada em `configs/busi.yaml`:

```bash
python main.py --config configs/busi.yaml validar
```

Arquivos de configuração podem ficar em subpastas. O campo opcional `projeto.raiz` define a raiz usada para resolver `original/`, `aumentado/`, `teste/`, `modelos/`, `resultados/` e checkpoints locais. O preset do BUSI usa `projeto.raiz: ..` para continuar apontando para a raiz do repositório.

Para outro dataset, altere `dados` e também os prompts em `difusao.prompts` conforme o domínio. Uma classe sem prompt específico em `por_classe` usa apenas o prompt comum.

Também ajuste `difusao.modelo` para o checkpoint/pasta Diffusers utilizado. O padrão aponta para `modelos_difusao/sdxl_base_1.0.safetensors`. `somente_arquivos_locais` vem como `false` para permitir que configurações auxiliares sejam obtidas na primeira execução; depois que estiverem em cache, pode ser alterado para `true`.

Nos exemplos das seções seguintes, os comandos usam o `config.yaml` genérico. Para executar o preset do BUSI, acrescente `--config configs/busi.yaml` **antes** do subcomando, por exemplo:

```bash
python main.py --config configs/busi.yaml preparar --dataset-zip dataset.zip --sobrescrever
```

## 3. Preparar os dados

A preparação aceita dois modos.

### Um único ZIP

```bash
python main.py preparar --dataset-zip dataset.zip --sobrescrever
```

O dataset completo é dividido de forma estratificada em desenvolvimento e teste; depois o desenvolvimento é dividido em treino e validação:

```text
dataset completo
├── desenvolvimento -> original/
│   ├── treino
│   └── validação
└── teste -> teste/
```

As proporções são definidas em:

```yaml
divisao:
  proporcao_teste: 0.15
  proporcao_treino: 0.85
  proporcao_validacao: 0.15
```

`proporcao_teste` vale sobre o dataset completo. `proporcao_treino` e `proporcao_validacao` valem sobre o conjunto de desenvolvimento.

### Desenvolvimento e teste em ZIPs separados

```bash
python main.py preparar --dataset-zip desenvolvimento.zip --teste-zip teste.zip --sobrescrever
```

Nesse modo, o primeiro ZIP é usado integralmente como desenvolvimento e o segundo integralmente como teste. Apenas o desenvolvimento é dividido em treino e validação.

Em ambos os modos, a divisão é estratificada por classe e reproduzível pela semente de `experimento.semente`. Imagem e máscaras associadas permanecem sempre na mesma partição.

A preparação cria `resultados/divisao_preparacao.csv` e copia o desenvolvimento real para `aumentado/`.

Opcionalmente, materialize treino e validação em pastas:

```bash
python main.py preparar --dataset-zip dataset.zip --exportar-divisao --sobrescrever
```

## 4. Validar a estrutura

```bash
python main.py validar
```

O comando informa quantidades por classe, imagens originais/sintéticas e amostras sem máscara. Uma classe não listada em `classes_sem_mascara` gera erro caso alguma imagem real não tenha máscara.

## 5. Gerar imagens sintéticas

Teste a seleção das imagens sem carregar o modelo:

```bash
python main.py difusao --simulacao
```

Geração real:

```bash
python main.py difusao
```

Somente imagens do subconjunto de treino são usadas como origem. Checkpoint, número de variações, `strength`, passos e prompts são definidos em `config.yaml`.

## 6. Gerar máscaras das sintéticas

```bash
python main.py mascaras
```

Para imagens cuja máscara de origem contém região de interesse, o pipeline aplica o watershed guiado. Quando a referência é vazia, a sintética recebe uma máscara vazia e o watershed não é aplicado.

Para salvar imagens intermediárias do processo:

```bash
python main.py mascaras --debug --limite 5
```

## 7. Treinar os dois modelos

```bash
python main.py treinar
```

O cenário original usa apenas o treino real. O cenário aumentado usa o mesmo treino real mais sintéticas derivadas exclusivamente dessas imagens. A validação real é idêntica nos dois cenários.

Checkpoints:

```text
modelos/unet_original.pt
modelos/unet_aumentado.pt
```

## 8. Avaliar

```bash
python main.py avaliar
```

A avaliação usa apenas `teste/` e gera, entre outros:

```text
resultados/metricas_por_imagem.csv
resultados/resumo_metricas.csv
resultados/comparacao_iou_f1.png
resultados/sobreposicoes/
```

O resumo contém:

- `macro_todas`: média das métricas por imagem em todo o teste;
- `macro_regiao_presente`: média apenas das imagens cuja máscara de referência contém região de interesse;
- `micro_pixels`: soma global de TP, TN, FP e FN antes do cálculo das métricas.

Na implementação binária, F1 e Dice são equivalentes.

## 9. Análise complementar

```bash
python scripts/analisar_resultados.py
```

O script agrega métricas por classe e gera gráficos comparativos e curvas de aprendizado conjuntas em `resultados/analise/`.

## 10. Executar o pipeline completo

Depois de preparar e validar os dados:

```bash
python main.py pipeline
```

Para um teste rápido de integração:

```bash
python main.py pipeline --limite-geracao 3 --limite-avaliacao 5
```

Os limites não devem ser usados em um experimento final.

## Principais diretórios

```text
original/          desenvolvimento real
  <classes>/
aumentado/         desenvolvimento real + sintéticas
  <classes>/
teste/             teste real
  <classes>/
modelos/            checkpoints treinados
modelos_difusao/    checkpoints de difusão
resultados/          métricas, gráficos e manifestos
scripts/             análise complementar
src/                 implementação do pipeline
```

## Testes

Localmente:

```bash
python -m unittest discover -s tests -v
python -m compileall main.py src scripts tests
```

## Reprodutibilidade

Para comparações válidas, mantenha a mesma semente e divisão entre os cenários, nunca gere sintéticas a partir de validação/teste e versione o `config.yaml` usado no experimento. Datasets, checkpoints e resultados pesados não devem ser versionados no repositório.

O pipeline tem finalidade experimental e de pesquisa e não constitui ferramenta de diagnóstico.

## Licença e citação

O código deste repositório é distribuído sob a licença MIT, disponível em `LICENSE`. Datasets, checkpoints e modelos de terceiros **não** são redistribuídos por este projeto e permanecem sujeitos às respectivas licenças e termos de uso.

Para trabalhos acadêmicos, o repositório inclui `CITATION.cff`.
