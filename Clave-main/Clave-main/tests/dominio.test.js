const Aluno = require('../src/dominio/Aluno.js');
const Professor = require('../src/dominio/Professor.js');
const Avaliacao = require('../src/dominio/Avaliacao.js');
const StatusAula = require('../src/dominio/StatusAula.js');
const Aula = require('../src/dominio/Aula.js');

//CAMADA DE DOMÍNIO
describe('Camada de Domínio - Plataforma Clave', () => {

  // Teste 1 — Criação e manipulação de Aluno
  test('deve criar um aluno e adicionar uma aula à lista de aulas agendadas', () => {
    const aluno = new Aluno(1, 'Ana', 'ana@email.com');
    const professor = new Professor(1, 'João', 'joao@email.com', 'Professor de Violão/Guitarra', 100);
    const aula = new Aula(1, professor, aluno, '2025-11-13T10:00:00', 120);

    aluno.adicionarAula(aula);

    expect(aluno.obterNome()).toBe('Ana');
    expect(aluno.obterAulasAgendadas().length).toBe(1);
    expect(aluno.obterAulasAgendadas()[0]).toBeInstanceOf(Aula);
  });

  // Teste 2 — Comportamento de status da Aula
  test('deve confirmar e cancelar uma aula corretamente', () => {
    const aluno = new Aluno(1, 'Ana', 'ana@email.com');
    const professor = new Professor(2, 'Carlos', 'carlos@email.com', 'Professor de Piano', 90);
    const aula = new Aula(2, professor, aluno, '2025-11-14T09:00:00', 80);

    // Aula começa com status SOLICITADA
    expect(aula.obterStatus()).toBe(StatusAula.SOLICITADA);

    // Confirmar a aula
    aula.confirmar();
    expect(aula.obterStatus()).toBe(StatusAula.CONFIRMADA);

    // Cancelar a aula depois de confirmada
    aula.cancelarPorAluno();
    expect(aula.obterStatus()).toBe(StatusAula.CANCELADA_PELO_ALUNO);
  });

  // Teste 3 — Validação de Avaliação
  test('deve lançar erro ao criar avaliação com nota fora do intervalo 1 a 5', () => {
    const professor = new Professor(3, 'José', 'jose@email.com', 'Professor de Flauta-Doce', 75);

    expect(() => new Avaliacao(1, 'Aluno', professor, 6, 'Excelente!'))
      .toThrow('A nota da avaliação deve ser um valor entre 1 e 5.');

    // Avaliação válida
    const avaliacao = new Avaliacao(2, 'Maria', professor, 5, 'Ótimo professor!');
    expect(avaliacao.obterNota()).toBe(5);
    expect(avaliacao.obterComentario()).toBe('Ótimo professor!');
  });

  // Teste 4 — Cálculo da média das avaliações do professor
  test('deve calcular corretamente a média das avaliações recebidas pelo professor', () => {
    const professor = new Professor(4, 'Marcos', 'marcos@email.com', 'Professor de Violão', 110);
    const avaliacao1 = new Avaliacao(1, 'Ana', professor, 4, 'Boa aula.');
    const avaliacao2 = new Avaliacao(2, 'Bruno', professor, 5, 'Excelente!');
    const avaliacao3 = new Avaliacao(3, 'Carla', professor, 3, 'Poderia melhorar.');

    professor.adicionarAvaliacao(avaliacao1);
    professor.adicionarAvaliacao(avaliacao2);
    professor.adicionarAvaliacao(avaliacao3);

    const media = professor.calcularNotaMedia();
    expect(media).toBeCloseTo(4.0);
  });

  // Teste 5 — Alterar nome do professor e aluno
  test('deve alterar o nome do professor e do aluno corretamente', () => {
    const professor = new Professor(5, 'Lucas', 'lucas@email.com', 'Professor de Guitarra', 100);
    const aluno = new Aluno(5, 'Pedro', 'pedro@email.com');

    professor.alterarNome('Lucas Silva');
    aluno.alterarNome('Pedro Souza');

    expect(professor.obterNome()).toBe('Lucas Silva');
    expect(aluno.obterNome()).toBe('Pedro Souza');
  });

});
