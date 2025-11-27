class Aluno {
    #id;
    #nome;
    #email;
    #aulasAgendadas;

    constructor(id, nome, email) {
        this.#id = id;
        this.#nome = nome;
        this.#email = email;
        this.#aulasAgendadas = [];
    }

    /**
     * Adiciona uma nova aula à lista de aulas do aluno.
     * @param {Aula} aula
     */
    adicionarAula(aula) {
        this.#aulasAgendadas.push(aula);
    }

    // --- Métodos de Acesso ---
    obterId() {
        return this.#id;
    }
    obterNome() {
        return this.#nome;
    }
    obterEmail(){
        return this.#email;
    }
    obterAulasAgendadas() {
        return this.#aulasAgendadas;
    }
    alterarNome(novoNome){
        this.#nome = novoNome;
    }
}

module.exports = Aluno;