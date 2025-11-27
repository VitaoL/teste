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

    confirmar() {
        if (this.#status !== StatusAula.SOLICITADA) {
            throw new Error('Ação inválida: Apenas aulas com status "SOLICITADA" podem ser confirmadas.');
        }
        this.#status = StatusAula.CONFIRMADA;
    }

    cancelarPorAluno() {
        const podeCancelar = [StatusAula.SOLICITADA, StatusAula.CONFIRMADA].includes(this.#status);
        if (!podeCancelar) {
            throw new Error('Ação inválida: Esta aula não pode mais ser cancelada.');
        }
        this.#status = StatusAula.CANCELADA_PELO_ALUNO;
    }

    obterStatus() { return this.#status; }
    obterProfessor() { return this.#professor; }
    obterAluno() { return this.#aluno; }
    obterId() { return this.#id; }
    obterDataHora() { return this.#dataHora; }
    obterValorAcordado() { return this.#valorAcordado; }
}

module.exports = Aula;
