# Clave: Plataforma de Aulas de Música - Camada de Domínio e Persistência

Este repositório contém a implementação das camadas de **Domínio** e **Persistência** para o projeto da plataforma **Clave**. O objetivo é demonstrar a modelagem de entidades de negócio, a implementação de suas regras e a comunicação com um banco de dados relacional.

## 1. Sobre o Nome: Clave

O nome **Clave** foi escolhido por seu profundo significado no universo musical. A clave (como a Clave de Sol ou de Fá) é o símbolo no início de uma partitura que define a altura das notas musicais. Ela é, literalmente, a "chave" para a leitura e a compreensão da música.

Para a nossa plataforma, o nome simboliza que o projeto é a **chave para conectar alunos e professores**, destravando o potencial musical de cada um e fornecendo o guia certo para a jornada de aprendizado.

## 2. Contexto Acadêmico

Este projeto foi desenvolvido como parte da avaliação da disciplina de Engenharia de Software II, do curso de Ciência da Computação da PUC Minas.

- **Instituição:** Pontifícia Universidade Católica de Minas Gerais (PUC Minas)
- **Curso:** Ciência da Computação
- **Disciplina:** Engenharia de Software II
- **Professor:** João Paulo Coelho Furtado

## 3. Como rodar o front-end e a API

1. **Pré-requisitos**
   - Node.js 18+
   - PostgreSQL (opcional — o front usa dados fictícios caso o banco não esteja configurado)

2. **Instale as dependências**
   ```bash
   npm install
   ```
   > Caso o registro npm corporativo bloqueie downloads, consulte o time de infraestrutura; o projeto também funciona em modo demo sem instalar pacotes extras.

3. **Configure o banco (opcional)**
   - Crie um banco local e ajuste o arquivo `.env` na raiz com:
     ```
     DB_PASSWORD=senha123
     DB_USER=admin
     DB_NAME=meubanco
     DB_HOST=127.0.0.1
     DB_PORT=5432
     ```
   - Rode os scripts `database/create_tables.sql` e `database/insert_data.sql` para popular as tabelas.

4. **Suba o servidor + front-end**
   ```bash
   npm start
   ```
   Acesse http://localhost:3000 para abrir a SPA. Se o banco não responder, a vitrine usa uma lista de professores fictícia, mas todas as interações (busca, filtros, agendamento, login/cadastro) continuam disponíveis via fallback local.

5. **APIs disponíveis**
   - `GET /api/professores` e `GET /api/professores/:id` para listar/abrir perfis.
   - `POST /api/register` e `POST /api/login` para cadastro/login (retornam fallback local se o banco não estiver ativo).
   - `POST /api/payment` para registrar o agendamento de aula.

## 4. Arquitetura e Tecnologias

O projeto segue uma arquitetura em camadas para garantir a separação de responsabilidades:

-   **Domínio (`src/dominio`):** Contém as classes que representam o coração do negócio (`Professor`, `Aluno`, `Aula`, etc.), com suas regras e lógicas internas.
-   **Persistência (`src/persistencia`):** Responsável por toda a comunicação com o banco de dados. Utiliza o padrão Repository para abstrair as operações de CRUD.

- **Tecnologias Principais:**
    - **Runtime:** Node.js
    - **Linguagem:** JavaScript (Classes ES6+)
    - **Banco de Dados:** PostgreSQL
    - **Driver de Conexão:** `pg`

## 4. Estrutura do Projeto

```
clave-projeto/
|
+-- database/
|   |-- create_tables.sql
|   `-- insert_data.sql
|
+-- src/
|   +-- dominio/
|   |   |-- Aluno.js, Professor.js, Aula.js, ...
|   |
|   `-- persistencia/
|       |-- AlunoRepositorio.js, ProfessorRepositorio.js, ...
|       `-- database.js
|
+-- teste.js         (Script de simulação e teste)
+-- package.json
+-- package-lock.json
`-- README.md
```

## 5. Modelo de Domínio Detalhado

A camada de domínio modela os conceitos de negócio como objetos, encapsulando dados e regras.

### `Professor.js` e `Aluno.js`
- **Responsabilidade:** Representam os atores principais do sistema. Eles contêm os dados de identificação e as listas de objetos com os quais se relacionam (um professor tem uma agenda de aulas e avaliações recebidas; um aluno tem suas aulas agendadas).
- **Métodos de Negócio (`Professor.js`):**
    - `calcularNotaMedia()`: Este método demonstra o encapsulamento de uma regra de negócio. Em vez de calcular a média do professor em uma camada externa, a própria classe `Professor` é responsável por essa lógica, garantindo que o cálculo seja feito de forma consistente em qualquer parte do sistema.

### `Aula.js`
- **Responsabilidade:** Atua como o "coração" do domínio, representando o evento central do sistema: o agendamento de uma aula. Ela conecta um `Professor` e um `Aluno` e gerencia o ciclo de vida desse compromisso.
- **Métodos de Negócio:**
    - `confirmar()` e `cancelarPorAluno()`: Estes métodos implementam uma **máquina de estados**. Eles garantem que o status de uma aula só possa ser alterado a partir de um estado válido (por exemplo, uma aula só pode ser confirmada se estiver `SOLICITADA`). Isso protege a integridade do objeto `Aula` e evita estados inconsistentes.

### `Avaliacao.js`
- **Responsabilidade:** Representa o feedback que um aluno fornece sobre um professor.
- **Métodos de Negócio:**
    - O **construtor** da classe implementa uma regra de validação crucial: ele dispara um erro se a nota fornecida não estiver no intervalo de 1 a 5. Isso é um exemplo de "invariante de domínio", uma regra que deve ser sempre verdadeira para que o objeto seja considerado válido, garantindo a integridade dos dados desde sua criação.

## 6. Análise do Script de Teste (`teste.js`)

O arquivo `teste.js` não é um teste unitário formal, mas sim um **script de simulação e integração**. Seu objetivo é verificar se as camadas de Domínio e Persistência estão funcionando corretamente em conjunto, executando um fluxo completo de operações no banco de dados.

O que cada etapa do script está testando:

1.  **Criação das Entidades (Teste do `CREATE`)**
    -   **O que faz:** Cria instâncias das classes `Aluno` e `Professor` e chama os métodos `salvar()` de seus respectivos repositórios.
    -   **O que testa:**
        -   Se as classes de domínio (`Aluno`, `Professor`) podem ser instanciadas corretamente.
        -   Se a conexão com o banco de dados é estabelecida (`Database.getPool()`).
        -   Se os métodos `salvar()` dos repositórios conseguem executar a query `INSERT` no banco de dados sem erros.
        -   Se o banco de dados retorna o ID do novo registro criado.

2.  **Leitura dos Dados (Teste do `READ`)**
    -   **O que faz:** Usa o ID retornado na etapa anterior para chamar o método `buscarPorId()`.
    -   **O que testa:**
        -   Se o método `buscarPorId()` consegue executar uma query `SELECT ... WHERE id = ...` e encontrar o dado recém-criado.
        -   Se a lógica de mapeamento (Data Mapper) dentro do repositório está convertendo corretamente a linha do banco de dados de volta para um objeto da classe `Aluno`.

3.  **Atualização de Dados (Teste do `UPDATE`)**
    -   **O que faz:** Altera uma propriedade do objeto `Aluno` em memória (usando o método `alterarNome()`) e depois chama o método `atualizar()` do repositório.
    -   **O que testa:**
        -   Se o método `atualizar()` consegue executar uma query `UPDATE` no registro correto.
        -   Verifica a consistência dos dados, buscando o mesmo objeto novamente e confirmando que a alteração foi persistida no banco.

4.  **Exclusão dos Dados (Teste do `DELETE`)**
    -   **O que faz:** Chama o método `deletar()` do repositório, passando o ID do aluno criado no teste.
    -   **O que testa:**
        -   Se o método `deletar()` executa a query `DELETE` com sucesso.
        -   A eficácia da exclusão, tentando buscar o aluno novamente e esperando um resultado `null`, confirmando que ele foi removido.

5.  **Gerenciamento da Conexão (`finally`)**
    -   **O que faz:** O bloco `finally` garante que o método `Database.closePool()` seja chamado, independentemente de os testes terem sucesso ou falha.
    -   **O que testa:**
        -   A capacidade do sistema de liberar recursos de forma segura, fechando o pool de conexões com o banco para evitar que conexões fiquem "presas" e consumindo memória.

## 7. Como Executar

**Pré-requisitos:**
-   [Node.js](https://nodejs.org/) instalado.
-   Um servidor [PostgreSQL](https://www.postgresql.org/) rodando localmente.

**Passos:**

1.  **Configuração do Banco de Dados:**
    -   Crie um banco de dados no PostgreSQL (ex: `clave_db`).
    -   Execute o script `database/create_tables.sql` para criar as tabelas.
    -   Execute o script `database/insert_data.sql` para popular o banco.

2.  **Configuração da Aplicação:**
    -   Clone este repositório.
    -   Na pasta raiz, instale as dependências: `npm install pg`
    -   **IMPORTANTE:** Abra o arquivo `src/persistencia/database.js` e, na seção de configuração da conexão, **altere o campo `password: 'sua_senha'` para a senha real do seu banco de dados PostgreSQL.**

3.  **Execução do Script:**
    -   Abra um terminal na pasta raiz do projeto.
    -   Execute o arquivo de simulação com o comando:
        ```bash
        node teste.js
        ```

## 8. Grupo

-   Ana Cristina Martins Silva
-   Letícia Azevedo Cota Barbosa
-   Lucas Gabriel Almeida Gomes
-   Pedro Henrique Gaioso de Oliveira
-   Victor Lustosa
