# Pipeline de segmentação binária com aumento generativo

Pipeline experimental para comparar dois cenários de treinamento de segmentação binária:

1. **original** — modelo treinado apenas com imagens reais;
2. **aumentado** — mesmo modelo treinado com as mesmas imagens reais mais imagens sintéticas geradas por `img2img`.

As máscaras das imagens sintéticas são produzidas por watershed guiado pela máscara da imagem real de origem. A avaliação final é feita em um conjunto de teste composto apenas por imagens reais.

A segmentação é sempre binária: fundo (0) vs região de interesse (1). As classes configuradas no dataset são categorias dos casos/imagens usadas para organização, estratificação, prompts e relatórios; elas não são classes de saída da U-Net.

A seed contida em `config.yaml` controla a divisão dos dados, a inicialização do treinamento e a geração por difusão de forma a tornar o experimento replicável na medida do possível. Entretanto, resultados numericamente idênticos entre máquinas não são garantidos, pois versões de PyTorch, CUDA, cuDNN, drivers, GPU e determinadas operações podem introduzir diferenças.

O pipeline foi desenvolvido de forma genérica e testado utilizando o **BUSI (Breast Ultrasound Images Dataset)**, de Al-Dhabyani et al. (2020), disponível na [página oficial do dataset da Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) e descrito no artigo *Dataset of breast ultrasound images* (DOI: 10.1016/j.dib.2019.104863). 

Para a geração das imagens sintéticas, foi utilizado o modelo de difusão **Stable Diffusion XL 1.0 Base (SDXL Base 1.0)**, carregado localmente a partir do checkpoint `sdxl_base_1.0.safetensors`. O pipeline utiliza a configuração correspondente à arquitetura `stabilityai/stable-diffusion-xl-base-1.0`.


## QuickStart

Depois de instalar as dependências, configurar as classes/prompts no YAML e colocar o checkpoint de difusão no caminho indicado por `difusao.modelo`, o experimento completo pode ser executado com poucos comandos.

### Configuração genérica

Com um único ZIP contendo todo o dataset, o comando `preparar` cria automaticamente o conjunto de desenvolvimento e o teste externo fixo conforme `divisao.proporcao_teste`:

```bash
python main.py preparar --dataset-zip <nome_do_dataset.zip> --sobrescrever
python main.py validar
python main.py pipeline
```

O comando `pipeline` executa, em sequência:

```text
1. difusão de todo o desenvolvimento
2. geração das máscaras das sintéticas
3. StratifiedKFold no desenvolvimento
4. treinamento dos modelos finais
5. avaliação no teste externo fixo
```

O `pipeline` **não executa a etapa `preparar`**, portanto a preparação deve ser feita ao menos uma vez antes da execução completa.

Se o desenvolvimento e o teste externo já estiverem separados em dois ZIPs:

```bash
python main.py preparar --dataset-zip <nome_do_desenvolvimento.zip> --teste-zip <nome_do_teste.zip> --sobrescrever
python main.py validar
python main.py pipeline
```

### BUSI

Para reproduzir o experimento com o preset do BUSI, acrescente `--config configs/busi.yaml` antes de cada subcomando:

```bash
python main.py --config configs/busi.yaml preparar --dataset-zip <Dataset_BUSI.zip> --sobrescrever
python main.py --config configs/busi.yaml validar
python main.py --config configs/busi.yaml pipeline
```

Para uma verificação rápida da integração sem executar os cinco folds completos, use o modo de depuração:

```bash
python main.py pipeline --modo-simples --limite-geracao 3 --limite-avaliacao 5
```

`--modo-simples` e os limites são destinados apenas à depuração e não devem ser usados nos resultados finais do experimento.

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

A preparação separa primeiro um conjunto de teste externo fixo. Esse conjunto não participa da difusão, da geração de máscaras, da validação cruzada nem da escolha da duração do treinamento final.

### Um único ZIP

```bash
python main.py preparar --dataset-zip dataset.zip --sobrescrever
```

O dataset é dividido de forma estratificada em:

```text
dataset completo
├── desenvolvimento -> original/
│   └── usado posteriormente no StratifiedKFold
└── teste externo fixo -> teste/
    └── usado somente na avaliação final
```

A proporção do teste é configurada em:

```yaml
divisao:
  proporcao_teste: 0.20
```

### Desenvolvimento e teste em ZIPs separados

```bash
python main.py preparar --dataset-zip desenvolvimento.zip --teste-zip teste.zip --sobrescrever
```

Nesse modo, o primeiro ZIP é usado integralmente como desenvolvimento e o segundo integralmente como teste externo.

Em ambos os modos, imagem e máscaras associadas permanecem sempre na mesma partição. A preparação cria `resultados/divisao_preparacao.csv`, contendo apenas as partições `desenvolvimento` e `teste`, e copia o desenvolvimento real para `aumentado/`.

Opcionalmente, uma divisão simples treino/validação pode ser materializada apenas para depuração:

```bash
python main.py preparar --dataset-zip dataset.zip --exportar-divisao --sobrescrever
```

Essa divisão simples **não** é usada no experimento principal com validação cruzada.

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

A difusão é executada uma única vez sobre todo o conjunto de desenvolvimento. Isso permite reutilizar as mesmas sintéticas em todos os folds. O conjunto `teste/` nunca é processado por difusão.

Checkpoint, número de variações, `strength`, passos e prompts são definidos em `config.yaml`.

## 6. Gerar máscaras das sintéticas

```bash
python main.py mascaras
```

Antes do cálculo do gradiente, a imagem sintética em tons de cinza recebe um filtro de mediana do OpenCV (`cv2.medianBlur`). Configure `watershed.mediana_kernel` no YAML: `7` (padrão) aplica uma janela 7 × 7, `5` aplica 5 × 5 e `0` desativa para comparação com o comportamento anterior. Valores ativos devem ser inteiros ímpares maiores ou iguais a 3.

A filtragem modifica apenas a entrada do gradiente, sem substituir a imagem sintética salva nem a máscara original usada para os marcadores. Com `--debug`, o painel inclui o antes/depois em cinza e salva as etapas intermediárias em `resultados/watershed/debug/`.

Para imagens cuja máscara de origem contém região de interesse, o pipeline aplica o watershed guiado. Quando a referência é vazia, a sintética recebe uma máscara vazia e o watershed não é aplicado.

```bash
python main.py mascaras --debug --limite 5
```

## 7. Validação cruzada e treinamento final

```bash
python main.py treinar
```

Por padrão, o comando executa **StratifiedKFold** somente sobre `original/` (o conjunto de desenvolvimento). O número de folds é configurado no YAML:

```yaml
validacao_cruzada:
  folds: 5
  embaralhar: true
  treinar_modelos_finais: true
```

Para cada fold são treinados dois modelos com a mesma partição de validação:

```text
Fold i
├── cenário original
│   └── originais do treino do fold
└── cenário aumentado
    └── mesmos originais + sintéticas derivadas somente dessas originais

Validação do fold
└── somente imagens originais
```

Uma sintética cuja imagem de origem esteja na validação daquele fold permanece no disco, mas é automaticamente excluída do treinamento. Assim, nenhuma versão derivada da validação vaza para o treino.

Os resultados de cada fold ficam em:

```text
resultados/validacao_cruzada/
├── folds.csv
├── treinamentos_por_fold.csv
├── metricas_por_fold.csv
├── resumo_validacao_cruzada.csv
├── comparacao_pareada_por_fold.csv
├── comparacao_macro_todas.png
├── comparacao_macro_regiao_presente.png
└── fold_XX/
```

Os checkpoints intermediários ficam em `modelos/validacao_cruzada/fold_XX/`.

Depois dos folds, o pipeline escolhe para cada cenário o número de épocas finais como a **mediana** da melhor época observada nos folds. Em seguida treina um modelo final usando todo o conjunto de desenvolvimento:

```text
modelo final original  -> 100% das imagens originais de desenvolvimento
modelo final aumentado -> 100% das originais + sintéticas correspondentes
```

Esses modelos finais são salvos em:

```text
modelos/unet_original.pt
modelos/unet_aumentado.pt
```

Para uma depuração rápida sem os cinco folds:

```bash
python main.py treinar --modo-simples
```

O modo simples usa `divisao.proporcao_treino` e `divisao.proporcao_validacao` e não deve substituir o protocolo final do experimento.

## 8. Avaliação final

```bash
python main.py avaliar
```

A avaliação usa apenas o conjunto fixo em `teste/`, composto por imagens originais que não participaram do treinamento nem da validação cruzada.

O comando organiza os artefatos finais em duas pastas:

```text
resultados/analise_modelo_final/
resultados/sobreposicoes_modelo_final/
```

A análise está integrada a `src/avaliacao.py`. Em `resultados/analise_modelo_final/` são gerados:

```text
comparacao_metricas_por_classe.png
comparacao_metricas_todas_imagens.png
comparacao_metricas_imagens_com_regiao.png
curvas_aprendizado_comparadas.png
metricas_por_imagem.csv
metricas_por_classe.csv
resumo_metricas.csv
```

`comparacao_metricas_imagens_com_regiao.png` usa o escopo `macro_regiao_presente`, isto é, considera apenas imagens cuja máscara de referência contém região de interesse. No BUSI, isso corresponde às imagens com lesão (benignas e malignas). As sobreposições visuais entre referência e predições dos dois modelos ficam separadas em `resultados/sobreposicoes_modelo_final/`.

O resumo contém:

- `macro_todas`: média das métricas por imagem em todo o teste;
- `macro_regiao_presente`: média apenas das imagens cuja máscara de referência contém região de interesse;
- `micro_pixels`: soma global de TP, TN, FP e FN antes do cálculo das métricas.

Na implementação binária, F1 e Dice são equivalentes.

## 9. Executar o pipeline completo

Depois de preparar e validar os dados:

```bash
python main.py pipeline
```

O fluxo principal passa a ser:

```text
preparar
  -> teste externo fixo + desenvolvimento
  -> difusão de todo o desenvolvimento
  -> máscaras watershed + mediana
  -> StratifiedKFold no desenvolvimento
  -> treino final em todo o desenvolvimento
  -> avaliação única no teste externo
  -> análise complementar automática
```

Para testar a integração sem validação cruzada completa:

```bash
python main.py pipeline --modo-simples --limite-geracao 3 --limite-avaliacao 5
```

Os limites e o modo simples não devem ser usados no experimento final.

## Principais diretórios

```text
original/                     desenvolvimento real
aumentado/                    desenvolvimento real + sintéticas
teste/                        teste externo fixo, somente real
modelos/                      modelos finais
modelos/validacao_cruzada/    checkpoints dos folds
modelos_difusao/              checkpoints de difusão
resultados/                    avaliação final e manifestos
resultados/validacao_cruzada/ resultados dos folds
resultados/analise/            agregações e gráficos da avaliação final
src/                          implementação do pipeline
```

## Testes

Localmente:

```bash
python -m unittest discover -s tests -v
python -m compileall main.py src tests
```

## Reprodutibilidade

Para comparações válidas, mantenha a mesma semente e os mesmos folds entre os cenários. As sintéticas podem ser pré-geradas para todo o desenvolvimento, mas uma sintética só pode entrar no treino quando sua imagem original também pertence ao treino daquele fold; o teste externo nunca é sintetizado nem usado no ajuste. Versione o `config.yaml` usado no experimento. Datasets, checkpoints e resultados pesados não devem ser versionados no repositório.

O pipeline tem finalidade experimental e de pesquisa e não constitui ferramenta de diagnóstico.

## Licença e citação

O código deste repositório é distribuído sob a licença MIT, disponível em `LICENSE`. Datasets, checkpoints e modelos de terceiros **não** são redistribuídos por este projeto e permanecem sujeitos às respectivas licenças e termos de uso.

Para trabalhos acadêmicos, o repositório inclui `CITATION.cff`.
