const StatusAula = require('./StatusAula');

class Aula {
    #id;
    #professor;
    #aluno;
    #dataHora;
    #status;
    #valorAcordado;

    constructor(id, professor, aluno, dataHora, valorAcordado) {
        this.#id = id;
        this.#professor = professor;
        this.#aluno = aluno;
        this.#dataHora = new Date(dataHora);
        this.#valorAcordado = valorAcordado;
        this.#status = StatusAula.SOLICITADA;
    }

    // --- Métodos de Negócio ---
    confirmar() {
        if (this.#status === StatusAula.SOLICITADA) {
            this.#status = StatusAula.CONFIRMADA;
            console.log('Aula confirmada com sucesso!');
        } else {
            console.error('Ação inválida: Apenas aulas com status "SOLICITADA" podem ser confirmadas.');
        }
    }

    cancelarPorAluno() {
        if (this.#status === StatusAula.SOLICITADA || this.#status === StatusAula.CONFIRMADA) {
            this.#status = StatusAula.CANCELADA_PELO_ALUNO;
            console.log('Aula cancelada pelo aluno.');
        } else {
            console.error('Ação inválida: Esta aula não pode mais ser cancelada.');
        }
    }

    // --- Métodos de Acesso ---
    obterStatus() {
        return this.#status;
    }
    obterProfessor() {
        return this.#professor;
    }
    obterAluno() {
        return this.#aluno;
    }
    obterId() {
        return this.#id;
    }

    obterDataHora() {
        return this.#dataHora;
    }

    obterValorAcordado() {
        return this.#valorAcordado;
    }

}

module.exports = Aula;